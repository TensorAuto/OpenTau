#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the cross-rank lockstep eval rollout protocol (sharded-param eval).

Under FSDP / DeepSpeed ZeRO-3 every policy forward all-gathers params per layer,
so eval rollouts must keep the number and order of forwards matched across ranks
(``_RolloutLockstep`` in ``opentau/scripts/eval.py``). These tests cover:

* the lockstep gating logic (``_make_rollout_lockstep``),
* the ``rollout()`` round protocol with a scripted (no-process-group) lockstep,
* ghost rollouts / ghost tasks for ranks whose task shard is smaller,
* a real 2-rank gloo end-to-end ``eval_policy_all`` with uneven task counts and
  uneven episode lengths (CPU, ``mp.spawn``),
* a real 2-rank DeepSpeed ZeRO-3 end-to-end run (gpu marker, needs 2 CUDA
  devices) including training steps before/after the eval to exercise the param
  coordinator's trace-invalidation self-heal.
"""

import contextlib
import datetime
import os
import socket
import time
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from accelerate import DistributedType
from torch import nn

from opentau.scripts.eval import (
    _drain_ghost_tasks,
    _ghost_rollout,
    _make_rollout_lockstep,
    _RolloutLockstep,
    rollout,
)

# ---------------------------------------------------------------------------
# Shared fakes (module-level so macOS/linux `spawn` can pickle the workers).
# ---------------------------------------------------------------------------


class _FakeVecEnv:
    """Minimal vec-env surface for rollout(): reset/step/call/num_envs.

    All ``num_envs`` sub-envs terminate together after ``done_at`` steps with
    ``is_success=True``, so per-rank rollout lengths are controlled exactly.
    ``check_env_attributes_and_types`` / ``add_envs_task`` are patched out in the
    tests (they isinstance-check for real gym vector envs).
    """

    def __init__(self, num_envs: int = 2, done_at: int = 3, max_steps: int = 10, state_dim: int = 4):
        self.num_envs = num_envs
        self.done_at = done_at
        self.max_steps = max_steps
        self.state_dim = state_dim
        self.steps = 0

    def _obs(self) -> dict:
        return {
            "pixels": {"camera0": np.zeros((self.num_envs, 8, 8, 3), dtype=np.uint8)},
            "agent_pos": np.zeros((self.num_envs, self.state_dim), dtype=np.float32),
        }

    def reset(self, seed=None):
        self.steps = 0
        return self._obs(), {}

    def step(self, actions):
        self.steps += 1
        terminated = np.full((self.num_envs,), self.steps >= self.done_at, dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        reward = np.ones((self.num_envs,), dtype=np.float32)
        info = {"is_success": terminated.copy()}
        return self._obs(), reward, terminated, truncated, info

    def call(self, name, *args):
        if name == "_max_episode_steps":
            return [self.max_steps] * self.num_envs
        raise AttributeError(name)


class _FakeQueuePolicy(nn.Module):
    """Mimics the shared action-queue cadence of every LIBERO-eval policy.

    A model forward fires on call 1 after ``reset()`` and then every
    ``n_action_steps`` calls (the pi05/pi0 ``max_delay=0`` cadence) — i.e. the
    forward count is a pure function of call-count-since-reset, which is the
    property the lockstep protocol relies on.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 3, n_action_steps: int = 4, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_calls = 0
        self.n_forwards = 0
        self.n_resets = 0
        self._queue_len = 0

    def reset(self):
        self.n_resets += 1
        self._queue_len = 0

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        self.n_calls += 1
        state = batch["state"]
        if self._queue_len == 0:
            self.n_forwards += 1
            # The obs may live on another device than the net (preprocess targets
            # the accelerator/auto device); mirror real policies' device handling.
            _ = self.net(state.float().to(next(self.parameters()).device))
            self._queue_len = self.n_action_steps
        self._queue_len -= 1
        return torch.zeros((state.shape[0], self.action_dim), dtype=torch.float32)


class _ScriptedLockstep:
    """Duck-typed ``_RolloutLockstep`` that scripts the global-round outcome.

    ``any_rank_running`` returns True while the local rank runs, plus
    ``extra_rounds`` additional sync-only rounds (emulating slower ranks), then
    False. No process group involved.
    """

    def __init__(self, extra_rounds: int = 0, template: dict | None = None):
        self.extra_rounds = extra_rounds
        self.observation_template = template
        self.sync_only_rounds = 0
        self.accelerator = types.SimpleNamespace(unwrap_model=lambda m: m)

    def any_rank_running(self, local_running: bool) -> bool:
        if local_running:
            return True
        if self.extra_rounds > 0:
            self.extra_rounds -= 1
            self.sync_only_rounds += 1
            return True
        return False

    def task_plan_over_ranks(self, n_local_tasks: int) -> tuple[int, int]:
        return n_local_tasks, n_local_tasks


def _spawn_bounded(worker, args, nprocs: int = 2, timeout_s: float = 300.0):
    """mp.spawn with a hard join timeout.

    A protocol regression where every rank keeps looping with *matched*
    collectives never trips the process-group op timeout (all ops complete), so
    a plain ``mp.spawn(join=True)`` would hang the pytest session until the CI
    job timeout. Bound it: on timeout, kill the workers and fail the test.
    """
    ctx = mp.spawn(worker, args=args, nprocs=nprocs, join=False)
    try:
        # join() returns False whenever it reaped SOME process but not all —
        # it is designed to be looped; child exceptions re-raise out of it.
        deadline = time.monotonic() + timeout_s
        while not ctx.join(timeout=max(0.1, deadline - time.monotonic())):
            if time.monotonic() >= deadline:
                pytest.fail(f"spawned workers still running after {timeout_s}s — lockstep protocol hang")
    finally:
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
        # Reap without masking the primary outcome (terminated children make
        # this cleanup join raise ProcessExitedException).
        with contextlib.suppress(Exception):
            ctx.join(timeout=30)


def _make_lockstep_cfg(state_dim: int = 4, batch_size: int = 2):
    """Minimal cfg for real rollout()/eval_policy paths with the fakes above."""
    return types.SimpleNamespace(
        resolution=(8, 8),
        max_state_dim=state_dim,
        num_cams=1,
        eval=types.SimpleNamespace(
            seed_list=None,
            goal_frames_dir=None,
            decorrelate_rank_seeds=False,
            batch_size=batch_size,
            recording_root=None,
        ),
        env=types.SimpleNamespace(task="fake_task"),
        policy=types.SimpleNamespace(eval_use_discrete_actions=False),
    )


def _patch_rollout_env_helpers(monkeypatch):
    """Bypass the gym-vector isinstance checks for the fake vec env."""
    monkeypatch.setattr("opentau.scripts.eval.check_env_attributes_and_types", lambda env: None)
    monkeypatch.setattr(
        "opentau.scripts.eval.add_envs_task",
        lambda env, obs: {**obs, "prompt": ["fake task"] * env.num_envs},
    )
    monkeypatch.setattr("opentau.scripts.eval.add_eval_metadata", lambda obs, cfg: obs)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# _make_rollout_lockstep gating
# ---------------------------------------------------------------------------


class TestMakeRolloutLockstep:
    @staticmethod
    def _acc(distributed_type, *, num_processes=2, zero_stage=None):
        acc = types.SimpleNamespace(
            distributed_type=distributed_type,
            num_processes=num_processes,
            device=torch.device("cpu"),
        )
        if zero_stage is not None:
            acc.deepspeed_plugin = types.SimpleNamespace(
                hf_ds_config=types.SimpleNamespace(config={"zero_optimization": {"stage": zero_stage}})
            )
        return acc

    def test_none_accelerator(self):
        assert _make_rollout_lockstep(None) is None

    def test_single_process(self):
        assert _make_rollout_lockstep(self._acc(DistributedType.FSDP, num_processes=1)) is None

    def test_replicated_params(self):
        assert _make_rollout_lockstep(self._acc(DistributedType.MULTI_GPU)) is None
        assert _make_rollout_lockstep(self._acc(DistributedType.DEEPSPEED, zero_stage=2)) is None

    def test_sharded_but_uninitialized_process_group(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
        assert _make_rollout_lockstep(self._acc(DistributedType.FSDP)) is None

    def test_sharded_initialized(self, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        for acc in (
            self._acc(DistributedType.FSDP),
            self._acc(DistributedType.DEEPSPEED, zero_stage=3),
        ):
            lockstep = _make_rollout_lockstep(acc)
            assert isinstance(lockstep, _RolloutLockstep)
            assert lockstep.accelerator is acc


# ---------------------------------------------------------------------------
# rollout() round protocol (single process, scripted lockstep)
# ---------------------------------------------------------------------------


class TestRolloutLockstepProtocol:
    def _run(self, monkeypatch, *, lockstep, done_at=3, max_steps=10):
        _patch_rollout_env_helpers(monkeypatch)
        env = _FakeVecEnv(done_at=done_at, max_steps=max_steps)
        policy = _FakeQueuePolicy()
        cfg = _make_lockstep_cfg()
        ret = rollout(env=env, policy=policy, cfg=cfg, lockstep=lockstep)
        return env, policy, ret

    def test_sync_only_forwards_after_local_done(self, monkeypatch):
        """Locally done at 3 steps; 4 scripted extra rounds -> 7 select_action
        calls, but metrics reflect only the 3 real env steps."""
        lockstep = _ScriptedLockstep(extra_rounds=4)
        env, policy, ret = self._run(monkeypatch, lockstep=lockstep)
        assert policy.n_calls == 7
        assert lockstep.sync_only_rounds == 4
        assert env.steps == 3  # env never stepped during sync-only rounds
        assert ret["action"].shape[1] == 3
        assert ret["done"].shape[1] == 3
        assert bool(ret["success"][:, -1].all())
        # The fully assembled policy input was recorded as the ghost template.
        assert lockstep.observation_template is not None
        assert "state" in lockstep.observation_template
        assert "prompt" in lockstep.observation_template

    def test_no_extra_rounds_matches_lockstep_none(self, monkeypatch):
        lockstep = _ScriptedLockstep(extra_rounds=0)
        _, policy_ls, ret_ls = self._run(monkeypatch, lockstep=lockstep)
        _, policy_none, ret_none = self._run(monkeypatch, lockstep=None)
        assert policy_ls.n_calls == policy_none.n_calls == 3
        assert policy_ls.n_forwards == policy_none.n_forwards == 1
        for key in ("action", "reward", "success", "done"):
            torch.testing.assert_close(ret_ls[key], ret_none[key])

    def test_forward_cadence_preserved_through_sync_rounds(self, monkeypatch):
        """Sync-only rounds keep popping/replenishing the queue on the same
        cadence: 9 total calls at n_action_steps=4 -> forwards at calls 1, 5, 9."""
        lockstep = _ScriptedLockstep(extra_rounds=6)
        _, policy, _ = self._run(monkeypatch, lockstep=lockstep)
        assert policy.n_calls == 9
        assert policy.n_forwards == 3


# ---------------------------------------------------------------------------
# Ghost rollouts / ghost tasks
# ---------------------------------------------------------------------------


class TestGhostRollout:
    def test_counts_resets_and_template_use(self):
        template = {"state": torch.zeros((2, 4), dtype=torch.bfloat16)}
        lockstep = _ScriptedLockstep(extra_rounds=5, template=template)
        policy = _FakeQueuePolicy()
        _ghost_rollout(policy, lockstep)
        assert policy.n_resets == 1
        assert policy.n_calls == 5
        assert policy.n_forwards == 2  # calls 1 and 5 at n_action_steps=4

    def test_raises_without_template(self):
        lockstep = _ScriptedLockstep(extra_rounds=1, template=None)
        with pytest.raises(RuntimeError, match="no observation template"):
            _ghost_rollout(_FakeQueuePolicy(), lockstep)


class TestDrainGhostTasks:
    @pytest.mark.parametrize(
        ("n_episodes", "num_envs", "seed_list", "expected_n_batches"),
        [
            (16, 16, None, 1),
            (17, 16, None, 2),
            (5, 2, None, 3),
            (999, 2, "1,2,3", 2),  # seed_list overrides n_episodes -> 3 seeds / 2 envs
        ],
    )
    def test_ghost_rollouts_match_eval_policy_batch_plan(
        self, monkeypatch, n_episodes, num_envs, seed_list, expected_n_batches
    ):
        calls = {"ghost": 0}
        monkeypatch.setattr(
            "opentau.scripts.eval._ghost_rollout",
            lambda policy, lockstep: calls.__setitem__("ghost", calls["ghost"] + 1),
        )
        cfg = _make_lockstep_cfg()
        cfg.eval.seed_list = seed_list
        _drain_ghost_tasks(
            policy=_FakeQueuePolicy(),
            cfg=cfg,
            lockstep=_ScriptedLockstep(),
            n_ghost_tasks=2,
            n_episodes=n_episodes,
            num_envs=num_envs,
        )
        assert calls["ghost"] == 2 * expected_n_batches

    def test_zero_ghost_tasks_noop(self, monkeypatch):
        def _boom(*args, **kwargs):
            raise AssertionError("should not run")

        monkeypatch.setattr("opentau.scripts.eval._ghost_rollout", _boom)
        _drain_ghost_tasks(
            policy=_FakeQueuePolicy(),
            cfg=_make_lockstep_cfg(),
            lockstep=_ScriptedLockstep(),
            n_ghost_tasks=0,
            n_episodes=16,
            num_envs=16,
        )

    def test_no_manifest_gathers_in_drain(self, monkeypatch, tmp_path):
        """Ghost tasks are pure rollouts: the multi-rank manifest gather happens
        exactly once per eval_policy_all (after the drain), never per ghost task."""

        def _boom(rows, cfg, acc):
            raise AssertionError("ghost drain must not gather the goal manifest")

        monkeypatch.setattr("opentau.scripts.eval._ghost_rollout", lambda policy, lockstep: None)
        monkeypatch.setattr("opentau.scripts.eval._gather_and_write_goal_manifest", _boom)
        cfg = _make_lockstep_cfg()
        cfg.eval.goal_frames_dir = str(tmp_path)
        _drain_ghost_tasks(
            policy=_FakeQueuePolicy(),
            cfg=cfg,
            lockstep=_ScriptedLockstep(),
            n_ghost_tasks=3,
            n_episodes=2,
            num_envs=2,
        )


class TestEvalPolicyAllLockstepGating:
    @staticmethod
    def _stub_eval_policy(monkeypatch):
        def fake_eval_policy(**kwargs):
            return {"per_episode": []}

        monkeypatch.setattr("opentau.scripts.eval.eval_policy", fake_eval_policy)

    def test_forces_sequential_under_lockstep(self, monkeypatch, caplog):
        from opentau.scripts.eval import eval_policy_all

        monkeypatch.setattr("opentau.scripts.eval._make_rollout_lockstep", lambda acc: _ScriptedLockstep())
        self._stub_eval_policy(monkeypatch)
        cfg = _make_lockstep_cfg()
        with caplog.at_level("WARNING"):
            eval_policy_all(
                {"g": {0: _FakeVecEnv(), 1: _FakeVecEnv()}},
                policy=_FakeQueuePolicy(),
                n_episodes=1,
                cfg=cfg,
                max_parallel_tasks=4,
            )
        assert "forcing max_parallel_tasks" in caplog.text

    def test_zero_task_rank_raises_synchronously(self, monkeypatch):
        """A rank plan of (min=0, max>0) must raise at the aligned plan point on
        every rank — a one-sided raise later would strand peers in a collective."""
        from opentau.scripts.eval import eval_policy_all

        lockstep = _ScriptedLockstep()
        lockstep.task_plan_over_ranks = lambda n: (0, 2)  # some other rank is empty
        monkeypatch.setattr("opentau.scripts.eval._make_rollout_lockstep", lambda acc: lockstep)
        self._stub_eval_policy(monkeypatch)
        with pytest.raises(ValueError, match="at least one\\s+eval task"):
            eval_policy_all(
                {"g": {0: _FakeVecEnv(), 1: _FakeVecEnv()}},
                policy=_FakeQueuePolicy(),
                n_episodes=1,
                cfg=_make_lockstep_cfg(),
                max_parallel_tasks=1,
            )

    def test_subgoal_generator_rejected_under_lockstep(self, monkeypatch):
        """Subgoal-conditioned eval is guarded off under sharding: the pi07-family
        subgoal branch is decided from per-rank subgoal availability, which would
        desync the sharded-param all-gathers inside a matched round."""
        from opentau.scripts.eval import eval_policy_all

        monkeypatch.setattr("opentau.scripts.eval._make_rollout_lockstep", lambda acc: _ScriptedLockstep())
        self._stub_eval_policy(monkeypatch)
        with pytest.raises(ValueError, match="[Ss]ubgoal-conditioned eval"):
            eval_policy_all(
                {"g": {0: _FakeVecEnv()}},
                policy=_FakeQueuePolicy(),
                n_episodes=1,
                cfg=_make_lockstep_cfg(),
                max_parallel_tasks=1,
                subgoal_generator=object(),
            )


# ---------------------------------------------------------------------------
# 2-rank gloo end-to-end (CPU): uneven task shards + uneven episode lengths.
# ---------------------------------------------------------------------------


def _install_lockstep_worker_patches(rank: int, world_size: int, device: torch.device, distributed_type):
    """In-process (spawned child) patches: fake accelerator + env-helper bypass."""
    import opentau.envs.utils as envs_utils_mod
    import opentau.scripts.eval as eval_mod

    fake_acc = types.SimpleNamespace(
        num_processes=world_size,
        process_index=rank,
        local_process_index=rank,
        is_main_process=rank == 0,
        device=device,
        distributed_type=distributed_type,
        unwrap_model=lambda m: m.module if hasattr(m, "module") else m,
    )
    if distributed_type == DistributedType.DEEPSPEED:
        fake_acc.deepspeed_plugin = types.SimpleNamespace(
            hf_ds_config=types.SimpleNamespace(config={"zero_optimization": {"stage": 3}})
        )
    eval_mod.get_proc_accelerator = lambda: fake_acc
    envs_utils_mod.get_proc_accelerator = lambda: fake_acc
    eval_mod.check_env_attributes_and_types = lambda env: None
    eval_mod.add_envs_task = lambda env, obs: {**obs, "prompt": ["fake task"] * env.num_envs}
    eval_mod.add_eval_metadata = lambda obs, cfg: obs
    return fake_acc


def _run_uneven_eval_and_check(
    rank: int, world_size: int, policy_for_eval, counting_policy, goal_frames_dir: str | None = None
):
    """Shared body of the gloo/NCCL workers: 2 tasks on rank 0 vs 1 on rank 1,
    with different episode lengths, then cross-rank forward-count assertions.

    Expected lockstep schedule (n_action_steps=4, forward on calls 1, 5, 9, ...
    per rollout): task slot 1 runs max(4, 13) = 13 global rounds -> 4 forwards;
    task slot 2 runs max(7, ghost) = 7 rounds -> 2 forwards; total 6 on EVERY
    rank. Without lockstep the per-rank totals genuinely diverge (rank 0:
    1 + 2 = 3 forwards, rank 1: 4), which under ZeRO-3 desyncs the sharded-param
    all-gathers.

    When ``goal_frames_dir`` is set, also exercises the deferred goal-frame
    manifest path: per-task rows ride back on the eval_policy info and
    eval_policy_all gathers/writes them exactly once per rank after the
    ghost-task drain — despite the 2-vs-1 task shard imbalance.
    """
    import opentau.scripts.eval as eval_mod

    if rank == 0:
        envs = {"g": {0: _FakeVecEnv(done_at=4, max_steps=15), 1: _FakeVecEnv(done_at=7, max_steps=15)}}
    else:
        envs = {"g": {0: _FakeVecEnv(done_at=13, max_steps=15)}}

    cfg = _make_lockstep_cfg()
    if goal_frames_dir is not None:
        cfg.eval.goal_frames_dir = goal_frames_dir

    info = eval_mod.eval_policy_all(
        envs,
        policy=policy_for_eval,
        n_episodes=2,
        cfg=cfg,
        max_episodes_rendered=0,
        videos_dir=None,
        start_seed=7,
        max_parallel_tasks=1,
    )

    counts = [None] * world_size
    torch.distributed.all_gather_object(counts, counting_policy.n_forwards)
    assert len(set(counts)) == 1, f"per-rank forward counts diverged: {counts}"
    assert counts[0] == 6, f"unexpected forward count: {counts}"

    n_local_episodes = 4 if rank == 0 else 2
    assert info["overall"]["n_episodes"] == n_local_episodes
    assert info["overall"]["pc_success"] == 100.0

    if goal_frames_dir is not None and rank == 0:
        # Every episode succeeds and all tasks share the fake task name, so the
        # (task, seed, decoder) dedup collapses the harvested rows to one per
        # seed (7 and 8), written once by the main process.
        manifest = (Path(goal_frames_dir) / "manifest.csv").read_text().strip().splitlines()
        assert len(manifest) == 3, f"expected header + 2 rows, got: {manifest}"


def _lockstep_gloo_worker(rank: int, world_size: int, port: int, goal_frames_dir: str):
    """Must be module-level so the `spawn` start method can pickle it. Asserts
    raise in-process; the parent's bounded join re-raises them."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=120)
    )
    try:
        _install_lockstep_worker_patches(rank, world_size, torch.device("cpu"), DistributedType.FSDP)
        policy = _FakeQueuePolicy()
        _run_uneven_eval_and_check(rank, world_size, policy, policy, goal_frames_dir=goal_frames_dir)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.slow
def test_eval_policy_all_lockstep_two_rank_gloo(tmp_path):
    """End-to-end lockstep eval over a real 2-rank gloo group (CPU): uneven task
    shards (2 vs 1) and uneven episode lengths still produce identical per-rank
    forward counts (no NCCL-style desync), per-rank metrics are unaffected, and
    the deferred goal-frame manifest is written exactly once."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")
    _spawn_bounded(_lockstep_gloo_worker, args=(2, _find_free_port(), str(tmp_path)))


# ---------------------------------------------------------------------------
# 2-rank DeepSpeed ZeRO-3 end-to-end (GPU): the real sharded-param case.
# ---------------------------------------------------------------------------


def _lockstep_zero3_worker(rank: int, world_size: int, port: int):
    """Real DeepSpeed ZeRO-3 engine + lockstep eval + training steps around it.

    Validates the three things the CPU tests cannot: (1) the per-layer sharded-
    param all-gathers stay matched across ranks through uneven rollouts (no NCCL
    hang), (2) select_action works through the ZeRO-3 fetch hooks on direct
    module calls, and (3) training resumes cleanly after the eval (the param
    coordinator's trace invalidation self-heals, including the cross-rank
    submodule-order asserts in reset_step).
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)

    import deepspeed

    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=300)
    )
    try:
        device = torch.device("cuda", rank)
        _install_lockstep_worker_patches(rank, world_size, device, DistributedType.DEEPSPEED)

        policy = _FakeQueuePolicy(hidden=256)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        ds_config = {
            "train_batch_size": world_size,
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "zero_optimization": {"stage": 3},
            "zero_allow_untested_optimizer": True,
        }
        engine, _, _, _ = deepspeed.initialize(model=policy, optimizer=optimizer, config=ds_config)
        module = engine.module
        assert any(hasattr(p, "ds_id") for p in module.parameters()), "params not ZeRO-3 sharded"

        def _train_step():
            x = torch.randn(1, 4, device=device)
            loss = engine(x).square().mean()
            engine.backward(loss)
            engine.step()

        # Two steps BEFORE eval: puts the ZeRO-3 param coordinator in
        # COMPLETE-trace mode, the realistic in-training-eval starting state.
        for _ in range(2):
            _train_step()

        _run_uneven_eval_and_check(rank, world_size, engine, policy)

        # Two steps AFTER eval: the coordinator must self-heal its trace (the
        # eval forwards deviate from the recorded training trace) without
        # tripping reset_step's cross-rank submodule-order asserts.
        module.train()
        for _ in range(2):
            _train_step()
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >= 2 CUDA devices")
def test_eval_policy_all_lockstep_two_rank_zero3():
    """End-to-end lockstep eval on a real 2-rank DeepSpeed ZeRO-3 engine (NCCL),
    with uneven task shards and episode lengths, sandwiched between real
    training steps. Without lockstep this configuration hangs at NCCL.

    NOTE: the nightly gpu_test.yml runner (g6.2xlarge) has a single L4, so this
    skips there — it runs on multi-GPU dev boxes and any >= 2-GPU runner (the
    regression g6.12xlarge has 4). The gloo test above covers the protocol logic
    in the gating CPU suite; this one additionally covers the real sharded-param
    all-gather matching, the ZeRO-3 fetch hooks on direct module calls, and the
    param-coordinator trace self-heal.
    """
    _spawn_bounded(_lockstep_zero3_worker, args=(2, _find_free_port()))
