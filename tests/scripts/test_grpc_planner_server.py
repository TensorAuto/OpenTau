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

"""Tests for the high-level-planner orchestration in the gRPC server.

These tests exercise the planner logic in ``RobotPolicyServicer`` only: policy
loading is patched out and the Gemini planner is replaced with a controllable
fake. CPU-only, no network.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from opentau.configs.deployment import PlannerConfig
from opentau.planner.gemini_er_planner import PlanResult
from opentau.scripts.grpc import robot_inference_pb2
from opentau.scripts.grpc.server import RobotPolicyServicer


class FakePlanner:
    """Controllable stand-in for GeminiERPlanner."""

    def __init__(self):
        self.calls: list[dict] = []
        self.results: list[PlanResult | Exception] = []
        self.release = threading.Event()
        self.release.set()  # non-blocking by default

    def queue(self, *results: PlanResult | Exception):
        self.results.extend(results)

    def plan(self, task, images, state=None, memory=""):
        self.calls.append({"task": task, "images": images, "state": state, "memory": memory})
        assert self.release.wait(timeout=10), "FakePlanner was never released"
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _plan(subtask: str, memory: str) -> PlanResult:
    return PlanResult(subtask=subtask, memory=memory, raw_text="{}", latency_s=0.0)


def _request(prompt: str = "make a sandwich") -> robot_inference_pb2.ObservationRequest:
    request = robot_inference_pb2.ObservationRequest(prompt=prompt)
    request.images.add(image_data=b"\xff\xd8fakejpeg", encoding="jpeg")
    request.robot_state.state.extend([0.1, 0.2, 0.3])
    return request


@pytest.fixture
def make_servicer():
    """Factory building a servicer with a patched policy path and fake planner."""
    servicers = []

    def _make(**planner_overrides) -> tuple[RobotPolicyServicer, FakePlanner]:
        planner_cfg = PlannerConfig(**planner_overrides)
        cfg = SimpleNamespace(planner=planner_cfg, policy=SimpleNamespace(type="pi05"))
        fake_planner = FakePlanner()
        with (
            patch.object(RobotPolicyServicer, "_load_policy"),
            patch("opentau.scripts.grpc.server.GeminiERPlanner", return_value=fake_planner),
        ):
            servicer = RobotPolicyServicer(cfg)
        servicers.append(servicer)
        return servicer, fake_planner

    yield _make
    for servicer in servicers:
        servicer.close()


class TestPlannerDisabled:
    def test_prompt_untouched_and_no_planner(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=False)
        assert servicer._planner is None
        assert servicer._planner_thread is None

        request = _request("raw task")
        servicer._apply_planner(request)

        assert request.prompt == "raw task"
        assert fake_planner.calls == []


class TestPlannerEnabled:
    def test_first_request_blocks_and_uses_subtask(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True)
        fake_planner.queue(_plan("pick up the bread", "started"))

        request = _request("make a sandwich")
        servicer._apply_planner(request)

        assert request.prompt == "pick up the bread"
        assert fake_planner.calls[0]["task"] == "make a sandwich"
        assert fake_planner.calls[0]["memory"] == ""
        assert fake_planner.calls[0]["images"] == [(b"\xff\xd8fakejpeg", "jpeg")]
        # Proto floats are float32, so compare approximately.
        assert fake_planner.calls[0]["state"] == pytest.approx([0.1, 0.2, 0.3])
        assert servicer._memory == "started"

    def test_first_plan_timeout_falls_back_to_raw_prompt(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, first_plan_timeout_s=0.1)
        fake_planner.release.clear()  # block the plan
        fake_planner.queue(_plan("subtask", "memory"))

        request = _request("raw task")
        servicer._apply_planner(request)
        assert request.prompt == "raw task"

        # Let the in-flight plan finish; its result must still be committed.
        fake_planner.release.set()
        _wait_until(lambda: servicer._current_subtask() == "subtask")
        request = _request("raw task")
        servicer._apply_planner(request)
        assert request.prompt == "subtask"

    def test_interval_gating(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, interval_s=100.0)
        fake_planner.queue(_plan("subtask", "memory"))

        for _ in range(3):
            request = _request()
            servicer._apply_planner(request)

        # Only the bootstrap plan ran; interval (100s) never elapsed again.
        assert len(fake_planner.calls) == 1
        assert request.prompt == "subtask"

    def test_replans_after_interval_with_updated_memory(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, interval_s=100.0)
        fake_planner.queue(_plan("first subtask", "first memory"), _plan("second subtask", "second memory"))

        servicer._apply_planner(_request())
        servicer._apply_planner(_request())  # caches a fresh observation
        # Short-circuit the 100s interval sleep instead of waiting it out.
        servicer._replan_event.set()
        _wait_until(lambda: len(fake_planner.calls) == 2)
        _wait_until(lambda: servicer._current_subtask() == "second subtask")

        assert fake_planner.calls[1]["memory"] == "first memory"
        request = _request()
        servicer._apply_planner(request)
        assert request.prompt == "second subtask"

    def test_requests_during_inflight_plan_do_not_stack_plans(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, first_plan_timeout_s=0.1)
        fake_planner.release.clear()
        fake_planner.queue(_plan("subtask", "memory"))

        servicer._apply_planner(_request())  # bootstrap, times out, plan in flight
        servicer._apply_planner(_request())
        servicer._apply_planner(_request())

        fake_planner.release.set()
        _wait_until(lambda: servicer._current_subtask() == "subtask")
        # The loop ran exactly one plan; the extra requests only refreshed the
        # cached observation (next replan comes after interval_s = 100s).
        assert len(fake_planner.calls) == 1

    def test_task_change_resets_memory_and_replans(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, interval_s=100.0)
        fake_planner.queue(_plan("subtask A", "memory A"), _plan("subtask B", "memory B"))

        request_a = _request("task A")
        servicer._apply_planner(request_a)
        assert request_a.prompt == "subtask A"

        request_b = _request("task B")
        servicer._apply_planner(request_b)
        assert request_b.prompt == "subtask B"
        assert fake_planner.calls[1]["task"] == "task B"
        assert fake_planner.calls[1]["memory"] == ""  # reset, not "memory A"

    def test_planner_exception_keeps_previous_subtask(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, interval_s=100.0)
        fake_planner.queue(_plan("good subtask", "memory"), RuntimeError("api down"))

        servicer._apply_planner(_request())
        servicer._apply_planner(_request())  # caches a fresh observation
        servicer._replan_event.set()  # short-circuit the interval sleep
        _wait_until(lambda: len(fake_planner.calls) == 2)

        request = _request()
        servicer._apply_planner(request)
        assert request.prompt == "good subtask"

    def test_stale_plan_does_not_commit_after_task_change(self, make_servicer):
        servicer, fake_planner = make_servicer(enabled=True, first_plan_timeout_s=0.1)
        fake_planner.release.clear()
        fake_planner.queue(_plan("stale subtask", "stale memory"), _plan("fresh subtask", "fresh memory"))

        servicer._apply_planner(_request("task A"))  # plan for A stuck in flight

        # Task changes while A's plan is still running.
        thread = threading.Thread(target=servicer._apply_planner, args=(_request("task B"),))
        thread.start()
        _wait_until(lambda: servicer._task == "task B")
        fake_planner.release.set()
        thread.join(timeout=10)
        assert not thread.is_alive()

        _wait_until(lambda: len(fake_planner.calls) == 2)
        _wait_until(lambda: servicer._current_subtask() == "fresh subtask")
        assert servicer._memory == "fresh memory"

    def test_get_action_chunk_uses_subtask_end_to_end(self, make_servicer):
        """GetActionChunk substitutes the subtask before preparing the observation."""
        servicer, fake_planner = make_servicer(enabled=True)
        fake_planner.queue(_plan("pick up the bread", "started"))

        servicer._prepare_observation = MagicMock(return_value=({}, None, None))
        servicer.policy = MagicMock()
        servicer.policy.sample_actions.return_value = torch.zeros((1, 2, 3))

        response = servicer.GetActionChunk(_request("make a sandwich"), MagicMock())

        prepared_request = servicer._prepare_observation.call_args.args[0]
        assert prepared_request.prompt == "pick up the bread"
        assert len(response.action_chunk) == 2


def _wait_until(predicate, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")
