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

import pytest
import torch
from transformers import AutoTokenizer

from opentau.configs.types import FeatureType, NormalizationMode, PolicyFeature
from opentau.policies.pi07_paligemma.low_level.configuration_pi07_low_level import (
    PI07PaligemmaLowLevelConfig,
)
from opentau.policies.pi07_paligemma.low_level.modeling_pi07_low_level import (
    ContextItem,
    PI07PaligemmaLowLevelFlowMatching,
    PI07PaligemmaLowLevelPolicy,
    make_att_2d_masks,
)


def _legacy_embed_prefix(
    model,
    *,
    videos,
    vid_masks,
    lang_tokens,
    lang_masks,
    state,
    response_tokens,
    response_masks,
    metadata_tokens,
    metadata_masks,
    subgoal_videos=(),
    subgoal_vid_masks=(),
    discrete_actions=None,
    discrete_action_masks=None,
    obs_history_is_pad=None,
    return_items=False,
):
    """Drive ``PI07PaligemmaLowLevelPolicy._build_prefix_items`` from the
    raw tensors these tests already construct, then run the resulting
    items through ``model.embed_prefix``.

    Going through the production layout method keeps the tests on a
    single source of truth: any change to ``_build_prefix_items``
    (the whole point of this refactor — adding/reordering blocks should
    only require editing one place) is exercised here automatically,
    rather than silently testing a parallel re-implementation.

    The wiring: a bare ``PI07PaligemmaLowLevelPolicy`` instance is
    fabricated with ``object.__new__`` and its ``prepare_*`` methods
    are stubbed to return the supplied tensors. The fake policy
    borrows the model's ``language_tokenizer`` so ``_embed_text``
    can resolve the layout strings.

    Contract: ``_build_prefix_items`` is expected to read only the
    ``language_tokenizer`` and ``prepare_*`` attributes wired below.
    To keep silent regressions visible — if a future change to
    ``_build_prefix_items`` reaches for some other attribute (e.g.
    ``self.config``) — we install a strict ``__class__`` whose
    ``__getattr__`` raises with a clear message naming the missing
    attribute, rather than letting Python fall back to a default
    ``AttributeError`` that points at an unrelated test failure.
    """

    class _StrictFakePolicy(PI07PaligemmaLowLevelPolicy):
        """Subclass that errors loudly on any unstubbed attribute access.

        Bypasses ``PI07PaligemmaLowLevelPolicy.__init__`` via
        ``object.__new__``; only the attributes assigned below are
        valid. Any other attribute read raises a clear error that
        names the missing attribute and the helper that owns it,
        rather than the bare ``AttributeError`` that would otherwise
        bubble up from deep inside ``_build_prefix_items``.
        """

        def __getattr__(self, name):  # noqa: D401 - clarity-first error
            raise AttributeError(
                f"_legacy_embed_prefix fake policy is missing attribute "
                f"{name!r}; if _build_prefix_items now reads this, extend "
                f"the helper in tests/policies/test_pi07_paligemma_low_level.py."
            )

    fake_policy = object.__new__(_StrictFakePolicy)
    fake_policy.language_tokenizer = model.language_tokenizer
    fake_policy.prepare_videos = lambda batch: (list(videos), list(vid_masks))
    fake_policy.prepare_language = lambda batch: (lang_tokens, lang_masks)
    fake_policy.prepare_response = lambda batch: (response_tokens, response_masks)
    fake_policy.prepare_metadata = lambda batch: (metadata_tokens, metadata_masks)
    fake_policy.prepare_subgoal_images = lambda batch: (
        list(subgoal_videos),
        list(subgoal_vid_masks),
    )
    fake_policy.prepare_state = lambda batch: state

    batch: dict = {}
    if obs_history_is_pad is not None:
        batch["obs_history_is_pad"] = obs_history_is_pad

    items = PI07PaligemmaLowLevelPolicy._build_prefix_items(
        fake_policy,
        batch,
        include_discrete_actions=discrete_actions is not None,
        discrete_actions=discrete_actions,
        discrete_action_masks=discrete_action_masks,
    )
    if return_items:
        return items
    embs, pad_masks, att_masks, _num_cross = PI07PaligemmaLowLevelFlowMatching.embed_prefix(model, items)
    return embs, pad_masks, att_masks


# Config defaults used across the test.
NUM_CAMERAS = 2
SIGLIP_TOKENS_PER_CAMERA = 256
NUM_SUBGOAL_CAMERAS = 1
SIGLIP_TOKENS_PER_SUBGOAL = 256
PROMPT_MAX_LENGTH = 256
RESPONSE_MAX_LENGTH = 52
METADATA_MAX_LENGTH = 52
DISCRETE_ACTION_MAX_LENGTH = 32
CHUNK_SIZE = 50
MAX_STATE_DIM = 32
MAX_ACTION_DIM = 32

# For training the state is provided as (B, n_obs_steps, D) so T = n_obs_steps.
N_OBS_STEPS = 8

VIDEO_TOKENS = NUM_CAMERAS * SIGLIP_TOKENS_PER_CAMERA  # 512
LANG_START = VIDEO_TOKENS  # 512
SUBGOAL_TOKENS = NUM_SUBGOAL_CAMERAS * SIGLIP_TOKENS_PER_SUBGOAL  # 256

# For inference: no discrete actions; state is 1 timestep for embed_prefix.
INFER_STATE_TOKENS = 1


class TestPI07PaligemmaLowLevelIntegration:
    """Integration tests for the PI07 low-level pipeline."""

    @staticmethod
    def _make_config() -> PI07PaligemmaLowLevelConfig:
        config = PI07PaligemmaLowLevelConfig(
            n_obs_steps=N_OBS_STEPS,
            chunk_size=CHUNK_SIZE,
            n_action_steps=CHUNK_SIZE,
            max_state_dim=MAX_STATE_DIM,
            max_action_dim=MAX_ACTION_DIM,
            prompt_max_length=PROMPT_MAX_LENGTH,
            response_max_length=RESPONSE_MAX_LENGTH,
            metadata_max_length=METADATA_MAX_LENGTH,
            discrete_action_max_length=DISCRETE_ACTION_MAX_LENGTH,
            normalization_mapping={
                "VISUAL": NormalizationMode.IDENTITY,
                "STATE": NormalizationMode.MIN_MAX,
                "ACTION": NormalizationMode.MEAN_STD,
            },
        )
        config.input_features = {
            "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(MAX_STATE_DIM,)),
        }
        config.output_features = {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(CHUNK_SIZE, MAX_ACTION_DIM)),
        }
        return config

    @staticmethod
    def _indicator_lens(tokenizer):
        """Fixed strings inserted by ``embed_prefix`` (matches modeling layout).

        With at least one optional middle block populated (as in this test),
        the state-end separator is ``", "`` and the trailing prefix-end is
        ``":\\n"``. With no optional content the state-end collapses to
        ``":\\n"`` and the prefix-end is omitted entirely.
        """
        return {
            "state_lead": len(tokenizer.encode("State: ", add_special_tokens=False)),
            "comma": len(tokenizer.encode(", ", add_special_tokens=False)),
            "subgoal_lead": len(tokenizer.encode("Subgoal: ", add_special_tokens=False)),
            "prefix_end": len(tokenizer.encode(":\n", add_special_tokens=False)),
            "action_lead": len(tokenizer.encode("Action: ", add_special_tokens=False)),
        }

    @classmethod
    def _train_prefix_total(cls, tokenizer) -> int:
        m = cls._indicator_lens(tokenizer)
        p = 0
        p += VIDEO_TOKENS
        p += PROMPT_MAX_LENGTH
        p += m["state_lead"] + N_OBS_STEPS
        p += m["comma"] + RESPONSE_MAX_LENGTH
        p += m["comma"] + METADATA_MAX_LENGTH
        p += m["comma"] + m["subgoal_lead"] + SUBGOAL_TOKENS
        p += m["prefix_end"]
        p += m["action_lead"] + DISCRETE_ACTION_MAX_LENGTH
        return p

    @classmethod
    def _infer_prefix_total(cls, tokenizer) -> int:
        m = cls._indicator_lens(tokenizer)
        p = 0
        p += VIDEO_TOKENS
        p += PROMPT_MAX_LENGTH
        p += m["state_lead"] + INFER_STATE_TOKENS
        p += m["comma"] + RESPONSE_MAX_LENGTH
        p += m["comma"] + METADATA_MAX_LENGTH
        p += m["comma"] + m["subgoal_lead"] + SUBGOAL_TOKENS
        p += m["prefix_end"]
        return p

    @staticmethod
    def _check_ones_before_zeros(mask_slice):
        """Check that in a 1-D boolean mask all Trues precede all Falses."""
        mask = mask_slice.cpu().numpy()
        first_zero = None
        for idx, val in enumerate(mask):
            if val == 0:
                first_zero = idx
                break
        if first_zero is not None:
            assert all(v == 0 for v in mask[first_zero:]), f"Zeros not contiguous: {mask}"
            assert all(v == 1 for v in mask[:first_zero]), f"Ones not contiguous: {mask}"
        else:
            assert all(v == 1 for v in mask), f"Expected all ones: {mask}"

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def _verify_pad_masks(self, prefix_pad_masks, suffix_pad_masks, tokenizer, inference_mode=False):
        assert prefix_pad_masks.shape[0] == 1
        total = self._infer_prefix_total(tokenizer) if inference_mode else self._train_prefix_total(tokenizer)
        assert prefix_pad_masks.shape[1] == total
        assert prefix_pad_masks.dtype == torch.bool
        assert suffix_pad_masks.shape == (1, CHUNK_SIZE)
        assert suffix_pad_masks.dtype == torch.bool

        m = self._indicator_lens(tokenizer)

        lang_slice = slice(LANG_START, LANG_START + PROMPT_MAX_LENGTH)
        state_t = INFER_STATE_TOKENS if inference_mode else N_OBS_STEPS

        resp_lo = LANG_START + PROMPT_MAX_LENGTH + m["state_lead"] + state_t + m["comma"]
        resp_slice = slice(resp_lo, resp_lo + RESPONSE_MAX_LENGTH)

        # Metadata precedes subgoal; subgoal (comma + "Subgoal: " + image
        # tokens) sits at the prefix tail.
        meta_lo = resp_lo + RESPONSE_MAX_LENGTH + m["comma"]
        meta_slice = slice(meta_lo, meta_lo + METADATA_MAX_LENGTH)

        sg_lo = meta_lo + METADATA_MAX_LENGTH + m["comma"] + m["subgoal_lead"]
        sg_slice = slice(sg_lo, sg_lo + SUBGOAL_TOKENS)

        for i in range(prefix_pad_masks.shape[0]):
            assert torch.all(prefix_pad_masks[i, :VIDEO_TOKENS] == 1)
            self._check_ones_before_zeros(prefix_pad_masks[i, lang_slice])
            self._check_ones_before_zeros(prefix_pad_masks[i, resp_slice])
            assert torch.all(prefix_pad_masks[i, sg_slice] == 1)
            self._check_ones_before_zeros(prefix_pad_masks[i, meta_slice])

            if not inference_mode:
                da_lo = sg_lo + SUBGOAL_TOKENS + m["prefix_end"] + m["action_lead"]
                da_slice = slice(da_lo, da_lo + DISCRETE_ACTION_MAX_LENGTH)
                self._check_ones_before_zeros(prefix_pad_masks[i, da_slice])

            self._check_ones_before_zeros(suffix_pad_masks[i])

    def _verify_position_ids(
        self,
        prefix_position_ids,
        suffix_position_ids,
        prefix_pad_masks,
        suffix_pad_masks,
        tokenizer,
        inference_mode=False,
    ):
        expected_prefix = torch.cumsum(prefix_pad_masks, dim=1) - 1
        assert torch.equal(prefix_position_ids, expected_prefix)

        if inference_mode:
            prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        else:
            prefix_offsets = torch.sum(prefix_pad_masks[:, :-DISCRETE_ACTION_MAX_LENGTH], dim=-1)[:, None]

        expected_suffix = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        assert torch.equal(suffix_position_ids, expected_suffix)

    def _verify_vlm_attention_mask(
        self, vlm_attention_mask, prefix_pad_masks, prefix_att_masks, inference_mode=False
    ):
        del inference_mode  # same rule as training
        expected = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        assert torch.equal(vlm_attention_mask, expected), (
            f"VLM attention mask mismatch vs make_att_2d_masks.\n"
            f"Diff indices: {(vlm_attention_mask != expected).nonzero(as_tuple=False)[:20]}"
        )

    def _verify_action_expert_attention_mask(
        self,
        action_expert_attention_mask,
        prefix_pad_masks,
        suffix_pad_masks,
        suffix_att_masks,
        inference_mode=False,
    ):
        if inference_mode:
            num_cross = prefix_pad_masks.shape[1]
        else:
            num_cross = prefix_pad_masks.shape[1] - DISCRETE_ACTION_MAX_LENGTH

        expected = make_att_2d_masks(
            suffix_pad_masks,
            suffix_att_masks,
            n_cross_att_tokens=num_cross,
            cross_att_pad_masks=prefix_pad_masks[:, :num_cross],
        )
        assert torch.equal(action_expert_attention_mask, expected), (
            f"Action expert attention mask mismatch vs make_att_2d_masks.\n"
            f"Diff indices: {(action_expert_attention_mask != expected).nonzero(as_tuple=False)[:20]}"
        )

    # ------------------------------------------------------------------
    # Main integration test
    # ------------------------------------------------------------------

    @pytest.mark.skip(reason="Requires too much memory, does not fit on RTX 3090 24GB")
    @pytest.mark.gpu
    @pytest.mark.slow
    def test_complete_pi07_low_level_pipeline(self, lerobot_dataset_metadata):
        """Test the PI07 low-level pipeline: forward (training) and select_action (inference)."""

        config = self._make_config()
        policy = PI07PaligemmaLowLevelPolicy(config, per_dataset_stats=[lerobot_dataset_metadata.stats])
        tokenizer = policy.model.language_tokenizer

        batch_size = 1
        batch = {
            "camera0": torch.randn(batch_size, N_OBS_STEPS, 3, 224, 224),
            "camera1": torch.randn(batch_size, N_OBS_STEPS, 3, 224, 224),
            "subgoal0": torch.randn(batch_size, 3, 224, 224),
            "state": torch.randn(batch_size, N_OBS_STEPS, MAX_STATE_DIM),
            "actions": torch.randn(batch_size, CHUNK_SIZE, MAX_ACTION_DIM),
            "prompt": ["Pick up the red block"],
            "response": ["Grasp the red block"],
            "speed": torch.tensor([50]),
            "quality": torch.tensor([3]),
            "mistake": torch.tensor([0]),
            "speed_is_pad": torch.tensor([False]),
            "quality_is_pad": torch.tensor([False]),
            "mistake_is_pad": torch.tensor([False]),
            "subgoal_is_pad": torch.tensor([False]),
            "action_is_pad": torch.cat(
                [
                    torch.zeros(batch_size, CHUNK_SIZE // 2, dtype=torch.bool),
                    torch.ones(batch_size, CHUNK_SIZE - CHUNK_SIZE // 2, dtype=torch.bool),
                ],
                dim=1,
            ),
            # Align SpaceTime encoder T with cameras/state; mark full history as real.
            "obs_history_is_pad": torch.zeros(batch_size, N_OBS_STEPS, dtype=torch.bool),
        }

        policy.to(dtype=torch.bfloat16, device="cuda")
        batch_cuda = {
            key: value.to("cuda", non_blocking=True, dtype=torch.bfloat16)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }
        batch_cuda["action_is_pad"] = batch_cuda["action_is_pad"].to(dtype=torch.bool)

        # ── Monkey-patch to capture intermediate tensors ──────────────
        captured = {}
        original_paligemma_forward = policy.model.paligemma_with_expert.forward
        original_embed_prefix = policy.model.embed_prefix
        original_embed_suffix = policy.model.embed_suffix

        def capture_forward(*args, **kwargs):
            if kwargs["inputs_embeds"][0] is not None:
                captured["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["vlm_position_ids"] = kwargs["position_ids"].clone()
            else:
                captured["action_expert_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured["action_expert_position_ids"] = kwargs["position_ids"].clone()
            return original_paligemma_forward(*args, **kwargs)

        def capture_embed_prefix(items, *args, **kwargs):
            # Truncate the discrete-action item to inject padding (same workaround as PI05 test).
            # ``items`` is the ``ContextItem`` list; the discrete-action block is the
            # trailing item with item_type == "discrete_action".
            half = DISCRETE_ACTION_MAX_LENGTH // 2
            for item in items:
                if item.item_type == "discrete_action":
                    da = item.data
                    dm = item.pad_mask
                    bsz = da.shape[0]
                    item.data = torch.cat(
                        (da[:, :half], torch.zeros((bsz, half), dtype=da.dtype, device=da.device)),
                        dim=-1,
                    )
                    item.pad_mask = torch.cat(
                        (dm[:, :half], torch.zeros((bsz, half), dtype=torch.bool, device=dm.device)),
                        dim=-1,
                    )
                    break
            result = original_embed_prefix(items, *args, **kwargs)
            captured["prefix_pad_masks"] = result[1].clone()
            captured["prefix_att_masks"] = result[2].clone()
            return result

        def capture_embed_suffix(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            captured["suffix_pad_masks"] = result[1].clone()
            captured["suffix_att_masks"] = result[2].clone()
            return result

        policy.model.paligemma_with_expert.forward = capture_forward
        policy.model.embed_prefix = capture_embed_prefix
        policy.model.embed_suffix = capture_embed_suffix

        # ── Training forward pass ────────────────────────────────────
        loss = policy.forward(batch_cuda)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.embed_suffix = original_embed_suffix

        # Verify normalize / unnormalize round-trip. Calling the submodules
        # directly bypasses the policy's `_resolve_dataset_index` helper, so
        # pass an explicit (B,) long index of zeros (this fixture is
        # single-dataset).
        action_output = {"actions": batch["actions"].to("cuda")}
        dataset_index = torch.zeros(batch_size, dtype=torch.long, device="cuda")
        assert torch.allclose(
            action_output["actions"],
            policy.unnormalize_outputs(policy.normalize_targets(action_output, dataset_index), dataset_index)[
                "actions"
            ],
            atol=1e-6,
        )

        # Verify all expected captures are present.
        for var in [
            "prefix_pad_masks",
            "prefix_att_masks",
            "suffix_pad_masks",
            "suffix_att_masks",
            "vlm_2d_attention_mask",
            "vlm_position_ids",
            "action_expert_2d_attention_mask",
            "action_expert_position_ids",
        ]:
            assert var in captured, f"{var} was not captured"

        assert captured["vlm_2d_attention_mask"].dtype == torch.bool
        assert captured["action_expert_2d_attention_mask"].dtype == torch.bool
        assert captured["prefix_pad_masks"].dtype == torch.bool
        assert captured["suffix_pad_masks"].dtype == torch.bool

        self._verify_pad_masks(captured["prefix_pad_masks"], captured["suffix_pad_masks"], tokenizer)
        self._verify_position_ids(
            captured["vlm_position_ids"],
            captured["action_expert_position_ids"],
            captured["prefix_pad_masks"],
            captured["suffix_pad_masks"],
            tokenizer,
        )
        self._verify_vlm_attention_mask(
            captured["vlm_2d_attention_mask"],
            captured["prefix_pad_masks"],
            captured["prefix_att_masks"],
        )
        self._verify_action_expert_attention_mask(
            captured["action_expert_2d_attention_mask"],
            captured["prefix_pad_masks"],
            captured["suffix_pad_masks"],
            captured["suffix_att_masks"],
        )

        assert isinstance(loss, dict)
        assert "MSE" in loss
        assert "CE" in loss
        assert all(v.isfinite() for v in loss.values() if torch.is_tensor(v))

        # Reset and check queue cleared.
        policy.reset()
        assert len(policy._action_queue) == 0

        # Optimizer params are non-empty.
        assert len(list(policy.get_optim_params())) > 0

        # ── Inference via select_action ──────────────────────────────
        captured_infer = {}

        def capture_forward_infer(*args, **kwargs):
            if kwargs["inputs_embeds"][0] is not None and kwargs.get("past_key_values") is None:
                captured_infer["vlm_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured_infer["vlm_position_ids"] = kwargs["position_ids"].clone()
            else:
                captured_infer["action_expert_2d_attention_mask"] = kwargs["attention_mask"].clone()
                captured_infer["action_expert_position_ids"] = kwargs["position_ids"].clone()
            return original_paligemma_forward(*args, **kwargs)

        def capture_embed_prefix_infer(*args, **kwargs):
            result = original_embed_prefix(*args, **kwargs)
            captured_infer["prefix_pad_masks"] = result[1].clone()
            captured_infer["prefix_att_masks"] = result[2].clone()
            return result

        def capture_embed_suffix_infer(*args, **kwargs):
            result = original_embed_suffix(*args, **kwargs)
            captured_infer["suffix_pad_masks"] = result[1].clone()
            captured_infer["suffix_att_masks"] = result[2].clone()
            return result

        policy.model.paligemma_with_expert.forward = capture_forward_infer
        policy.model.embed_prefix = capture_embed_prefix_infer
        policy.model.embed_suffix = capture_embed_suffix_infer

        # Inference batch: state is 2-D (B, D), images are 4-D (B, C, H, W).
        infer_batch = {
            "camera0": batch_cuda["camera0"][:, 0],  # (B, C, H, W)
            "camera1": batch_cuda["camera1"][:, 0],
            "subgoal0": batch_cuda["subgoal0"],  # already (B, C, H, W)
            "state": batch_cuda["state"][:, 0],  # (B, D)
            "prompt": ["Pick up the red block"],
            "response": ["Grasp the red block"],
            "speed": torch.tensor([50], device="cuda"),
            "quality": torch.tensor([3], device="cuda"),
            "mistake": torch.tensor([0], device="cuda"),
            "speed_is_pad": torch.tensor([False], device="cuda"),
            "quality_is_pad": torch.tensor([False], device="cuda"),
            "mistake_is_pad": torch.tensor([False], device="cuda"),
            "subgoal_is_pad": torch.tensor([False], device="cuda"),
        }
        action = policy.select_action(infer_batch)

        # Restore originals.
        policy.model.paligemma_with_expert.forward = original_paligemma_forward
        policy.model.embed_prefix = original_embed_prefix
        policy.model.embed_suffix = original_embed_suffix

        for var in [
            "prefix_pad_masks",
            "prefix_att_masks",
            "suffix_pad_masks",
            "suffix_att_masks",
            "vlm_2d_attention_mask",
            "vlm_position_ids",
            "action_expert_2d_attention_mask",
            "action_expert_position_ids",
        ]:
            assert var in captured_infer, f"{var} was not captured for select_action"

        assert captured_infer["vlm_2d_attention_mask"].dtype == torch.bool
        assert captured_infer["action_expert_2d_attention_mask"].dtype == torch.bool
        assert captured_infer["prefix_pad_masks"].dtype == torch.bool
        assert captured_infer["suffix_pad_masks"].dtype == torch.bool

        self._verify_pad_masks(
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            tokenizer,
            inference_mode=True,
        )
        self._verify_position_ids(
            captured_infer["vlm_position_ids"],
            captured_infer["action_expert_position_ids"],
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            tokenizer,
            inference_mode=True,
        )
        self._verify_vlm_attention_mask(
            captured_infer["vlm_2d_attention_mask"],
            captured_infer["prefix_pad_masks"],
            captured_infer["prefix_att_masks"],
            inference_mode=True,
        )
        self._verify_action_expert_attention_mask(
            captured_infer["action_expert_2d_attention_mask"],
            captured_infer["prefix_pad_masks"],
            captured_infer["suffix_pad_masks"],
            captured_infer["suffix_att_masks"],
            inference_mode=True,
        )

        assert action.shape == (1, MAX_ACTION_DIM)


class TestPI07PaligemmaLowLevelObsHistoryRegression:
    """GPU integration test for the SpaceTime SigLIP pipeline driven through a
    real PaliGemma backbone. The standalone (CPU-only) tests for the encoder's
    temporal-attention mask construction and PE behavior live in
    ``tests/policies/test_pi07_video_encoder_cpu.py`` — that file is the
    canonical home for ``SpaceTimeSiglipVideoEncoder`` unit tests, parametrized
    over both Gemma 3 and PaliGemma backbone projectors.
    """

    @staticmethod
    def _make_encoder(*, max_num_frames: int, spacetime_layer_stride: int = 4):
        """Build a SpaceTimeSiglipVideoEncoder without loading the full policy."""
        from opentau.policies.pi05.paligemma_with_expert import (
            PaliGemmaWithExpertConfig,
            PaliGemmaWithExpertModel,
        )
        from opentau.policies.pi07.video_encoder import SpaceTimeSiglipVideoEncoder

        paligemma_cfg = PaliGemmaWithExpertConfig(
            load_pretrained_paligemma=False,
            freeze_vision_encoder=True,
            discrete_action_vocab_size=256,
        )
        paligemma = PaliGemmaWithExpertModel(paligemma_cfg)
        paligemma = paligemma.to(device="cuda", dtype=torch.bfloat16)
        encoder = SpaceTimeSiglipVideoEncoder(
            vision_tower=paligemma.paligemma.vision_tower,
            multi_modal_projector=paligemma.paligemma.multi_modal_projector,
            max_num_frames=max_num_frames,
            spacetime_layer_stride=spacetime_layer_stride,
        )
        return encoder

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_single_frame_shape_and_mask_true(self):
        """T=1: output shape is (B, num_tokens, H), vid_mask all-True.

        obs_history_is_pad is irrelevant — True, False, and None all produce
        identical outputs (temporal attention is short-circuited).
        """
        bsz, t, c, h, w = 2, 1, 3, 224, 224
        encoder = self._make_encoder(max_num_frames=1)
        video = torch.rand(bsz, t, c, h, w, device="cuda", dtype=torch.bfloat16)

        out_none = encoder(video, obs_history_is_pad=None)
        vlm_hidden = encoder.multi_modal_projector.linear.out_features
        assert out_none.shape == (bsz, encoder.num_video_tokens, vlm_hidden)

        out_true = encoder(video, obs_history_is_pad=torch.ones(bsz, t, dtype=torch.bool, device="cuda"))
        out_false = encoder(video, obs_history_is_pad=torch.zeros(bsz, t, dtype=torch.bool, device="cuda"))

        torch.testing.assert_close(out_none, out_true, msg="T=1: obs_history_is_pad=True should match None")
        torch.testing.assert_close(out_none, out_false, msg="T=1: obs_history_is_pad=False should match None")


class TestPI07PaligemmaLowLevelStateEmbedding:
    """CPU-only tests verifying state embeddings in ``embed_prefix`` under all
    five observation-history conditions.

    For each case the test asserts:
      - state tokens appear in the prefix with the correct shape
      - the state pad_mask follows ``obs_history_is_pad`` semantics
      - state embeddings are non-zero (``state_proj`` actually ran)

    A separate pair of tests drives the core ``forward`` and ``sample_actions``
    paths through lightweight mocks to confirm they produce state embeddings
    (i.e. the ``state`` kwarg reaches ``embed_prefix``).
    """

    _STATE_LEAD_IDS = [10, 11, 12]  # "State: "
    _COMMA_IDS = [20]  # ", "
    _COLON_NL_IDS = [30]  # ":\n"
    _SUBGOAL_LEAD_IDS = [40, 41]  # "Subgoal: "
    _ACTION_LEAD_IDS = [50, 51]  # "Action: "

    @classmethod
    def _fake_tokenizer(cls):
        ids_by_str = {
            "State: ": cls._STATE_LEAD_IDS,
            ", ": cls._COMMA_IDS,
            ":\n": cls._COLON_NL_IDS,
            "Subgoal: ": cls._SUBGOAL_LEAD_IDS,
            "Action: ": cls._ACTION_LEAD_IDS,
        }

        class _FakeTokenizer:
            @staticmethod
            def encode(s, add_special_tokens=False):
                assert add_special_tokens is False
                return ids_by_str[s]

        return _FakeTokenizer()

    @classmethod
    def _make_mock_model(cls, hidden_size: int = 8):
        """Minimal ``PI07PaligemmaLowLevelFlowMatching`` with stubs — enough to
        drive ``embed_prefix`` on CPU with non-zero state embeddings."""
        h = hidden_size
        model = object.__new__(PI07PaligemmaLowLevelFlowMatching)

        # num_image_tokens mirrors embed_image's 4-token output below, so the
        # image-tower skip path (zeros of that width) matches the run path.
        _text_cfg = type("_TextConfig", (), {"hidden_size": h, "num_image_tokens": 4})()
        _pg_cfg = type("_PaliGemmaConfig", (), {"text_config": _text_cfg})()
        _stub_cfg = type("_StubConfig", (), {"paligemma_config": _pg_cfg})()

        class _PaliGemmaStub:
            config = _stub_cfg

            @staticmethod
            def embed_language_tokens(tokens):
                return torch.zeros(tokens.shape[0], tokens.shape[1], h, dtype=torch.bfloat16)

            @staticmethod
            def embed_image(img):
                return torch.zeros(img.shape[0], 4, h, dtype=torch.bfloat16)

            @staticmethod
            def embed_discrete_actions(tokens):
                return torch.zeros(tokens.shape[0], tokens.shape[1], h, dtype=torch.bfloat16)

        _video_enc = type(
            "_VideoEncoderStub",
            (),
            {"num_video_tokens": 6, "num_frames": 1},
        )()

        model.paligemma_with_expert = _PaliGemmaStub()
        model.video_encoder = _video_enc
        model.language_tokenizer = cls._fake_tokenizer()
        _real_proj = torch.nn.Linear(8, h)
        torch.nn.init.ones_(_real_proj.weight)
        torch.nn.init.zeros_(_real_proj.bias)
        model.state_proj = lambda x: _real_proj(x.float()).to(torch.bfloat16)

        def _embed_video(video, obs_history_is_pad=None):
            t = video.shape[1]
            max_t = model.video_encoder.num_frames
            if not 1 <= t <= max_t:
                raise ValueError(f"Expected 1 <= T <= {max_t}; got {t}.")
            return torch.zeros(video.shape[0], 6, h, dtype=torch.bfloat16)

        model.embed_video = _embed_video
        return model

    @classmethod
    def _call_embed_prefix(
        cls,
        *,
        n_obs_steps: int,
        obs_history_is_pad=None,
        bsize: int = 1,
        state_dim: int = 8,
        hidden_size: int = 8,
    ):
        """Helper that calls ``embed_prefix`` and returns (embs, pad_masks, att_masks)
        plus metadata dict for locating the state slice."""
        model = cls._make_mock_model(hidden_size=hidden_size)
        model.video_encoder.num_frames = n_obs_steps
        prompt_len = 5
        response_len = 4
        metadata_len = 3

        videos = [torch.zeros(bsize, n_obs_steps, 3, 8, 8)]
        vid_masks = [torch.ones(bsize, dtype=torch.bool)]
        lang_tokens = torch.zeros(bsize, prompt_len, dtype=torch.long)
        lang_masks = torch.ones(bsize, prompt_len, dtype=torch.bool)
        state = torch.randn(bsize, n_obs_steps, state_dim)
        response_tokens = torch.zeros(bsize, response_len, dtype=torch.long)
        response_masks = torch.zeros(bsize, response_len, dtype=torch.bool)
        metadata_tokens = torch.zeros(bsize, metadata_len, dtype=torch.long)
        metadata_masks = torch.zeros(bsize, metadata_len, dtype=torch.bool)

        (embs, pad_masks, att_masks) = _legacy_embed_prefix(
            model,
            videos=videos,
            vid_masks=vid_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            response_tokens=response_tokens,
            response_masks=response_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            obs_history_is_pad=obs_history_is_pad,
        )

        vid_tokens = 6
        state_lead = len(cls._STATE_LEAD_IDS)
        state_start = vid_tokens + prompt_len + state_lead
        state_end = state_start + n_obs_steps

        return (embs, pad_masks, att_masks), {
            "state_start": state_start,
            "state_end": state_end,
            "n_obs_steps": n_obs_steps,
            "bsize": bsize,
            "hidden_size": hidden_size,
        }

    # ------------------------------------------------------------------ #
    # Case 1: T=1, single frame — state mask always True for current step
    # ------------------------------------------------------------------ #
    def test_state_embedding_single_frame(self):
        """T=1: state has exactly 1 token, always unmasked (current step)."""
        (embs, pad_masks, _att), meta = self._call_embed_prefix(n_obs_steps=1)

        state_slice = embs[:, meta["state_start"] : meta["state_end"]]
        state_mask = pad_masks[:, meta["state_start"] : meta["state_end"]]

        assert state_slice.shape == (1, 1, meta["hidden_size"])
        assert state_mask.shape == (1, 1)
        assert torch.all(state_mask), "single-frame state must be unmasked"
        assert not torch.all(state_slice == 0), "state embedding should be non-zero"

    def test_state_embedding_single_frame_pad_irrelevant(self):
        """T=1: obs_history_is_pad=True/False/None all produce identical state masks."""
        results = {}
        for label, pad in [
            ("none", None),
            ("true", torch.ones(1, 1, dtype=torch.bool)),
            ("false", torch.zeros(1, 1, dtype=torch.bool)),
        ]:
            (_, pm, _), meta = self._call_embed_prefix(n_obs_steps=1, obs_history_is_pad=pad)
            results[label] = pm[:, meta["state_start"] : meta["state_end"]]

        assert torch.equal(results["none"], results["true"])
        assert torch.equal(results["none"], results["false"])
        assert torch.all(results["none"]), "state mask should always be True at T=1"

    # ------------------------------------------------------------------ #
    # Case 2: T>1, all history padded — only current step unmasked
    # ------------------------------------------------------------------ #
    def test_state_embedding_all_history_padded(self):
        """T>1 with [True, ..., True, False]: only the last state token is unmasked."""
        t = 4
        pad = torch.ones(1, t, dtype=torch.bool)
        pad[:, -1] = False

        (embs, pad_masks, _att), meta = self._call_embed_prefix(n_obs_steps=t, obs_history_is_pad=pad)

        state_slice = embs[:, meta["state_start"] : meta["state_end"]]
        state_mask = pad_masks[:, meta["state_start"] : meta["state_end"]]

        assert state_slice.shape == (1, t, meta["hidden_size"])
        assert state_mask.shape == (1, t)
        assert torch.all(~state_mask[:, :-1]), "padded history state tokens must be masked"
        assert torch.all(state_mask[:, -1]), "current state token must be unmasked"
        assert not torch.all(state_slice[:, -1] == 0), "current state embedding should be non-zero"

    def test_masked_history_state_zeroed_before_projection(self):
        """Defense-in-depth: with history padded, the masked state steps are
        zeroed *after* normalization and *before* ``state_proj``. The stub
        projection has zero bias, so ``proj(0) == 0`` surfaces the masked
        (historical) tokens as exactly-zero state embeddings while the current
        token stays non-zero. Keeps dropped history from leaking even if the
        attention mask regresses, and avoids the ``-mean/std`` that zeroing a
        raw state *before* normalization would produce.
        """
        t = 4
        pad = torch.ones(1, t, dtype=torch.bool)  # all-True: history_state_drop fired

        (embs, _pad_masks, _att), meta = self._call_embed_prefix(n_obs_steps=t, obs_history_is_pad=pad)

        state_slice = embs[:, meta["state_start"] : meta["state_end"]]
        assert state_slice.shape == (1, t, meta["hidden_size"])
        # Zero-bias stub proj: masked (historical) steps surface as exactly zero.
        assert torch.all(state_slice[:, :-1] == 0), "masked history state must be zeroed pre-projection"
        assert not torch.all(state_slice[:, -1] == 0), "current state embedding should be non-zero"

    def test_state_embedding_all_pad_true_matches_history_padded(self):
        """T>1 with all-True pad [True, True, True, True]: the last timestep is
        still forced unmasked by ``state_mask[:, -1] = True``, so the state mask
        and embeddings must be identical to the [True, ..., True, False] case."""
        t = 4
        bsize = 1
        state_dim = 8
        hidden_size = 8
        prompt_len = 5

        model = self._make_mock_model(hidden_size=hidden_size)
        model.video_encoder.num_frames = t
        state = torch.randn(bsize, t, state_dim)
        videos = [torch.zeros(bsize, t, 3, 8, 8)]
        vid_masks = [torch.ones(bsize, dtype=torch.bool)]
        lang_tokens = torch.zeros(bsize, prompt_len, dtype=torch.long)
        lang_masks = torch.ones(bsize, prompt_len, dtype=torch.bool)
        response_tokens = torch.zeros(bsize, 4, dtype=torch.long)
        response_masks = torch.zeros(bsize, 4, dtype=torch.bool)
        metadata_tokens = torch.zeros(bsize, 3, dtype=torch.long)
        metadata_masks = torch.zeros(bsize, 3, dtype=torch.bool)

        common_kwargs = {
            "videos": videos,
            "vid_masks": vid_masks,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
            "response_tokens": response_tokens,
            "response_masks": response_masks,
            "metadata_tokens": metadata_tokens,
            "metadata_masks": metadata_masks,
        }

        pad_with_current = torch.ones(bsize, t, dtype=torch.bool)
        pad_with_current[:, -1] = False
        (embs_partial, pm_partial, _) = _legacy_embed_prefix(
            model,
            **common_kwargs,
            obs_history_is_pad=pad_with_current,
        )

        pad_all_true = torch.ones(bsize, t, dtype=torch.bool)
        (embs_all, pm_all, _) = _legacy_embed_prefix(
            model,
            **common_kwargs,
            obs_history_is_pad=pad_all_true,
        )

        vid_tokens = 6
        state_lead = len(self._STATE_LEAD_IDS)
        state_start = vid_tokens + prompt_len + state_lead
        state_end = state_start + t

        state_mask_partial = pm_partial[:, state_start:state_end]
        state_mask_all = pm_all[:, state_start:state_end]
        assert torch.equal(state_mask_partial, state_mask_all), (
            "all-True pad should produce the same state mask as [T,T,T,F] "
            "because state_mask[:, -1] is always forced True"
        )

        state_emb_partial = embs_partial[:, state_start:state_end]
        state_emb_all = embs_all[:, state_start:state_end]
        torch.testing.assert_close(
            state_emb_partial,
            state_emb_all,
            msg="state embeddings should be identical regardless of current-frame pad value",
        )

    # ------------------------------------------------------------------ #
    # Case 3: T>1, all valid — every state token unmasked
    # ------------------------------------------------------------------ #
    def test_state_embedding_all_valid(self):
        """T>1 with all-False pad: every state token is unmasked."""
        t = 4
        no_pad = torch.zeros(1, t, dtype=torch.bool)

        (embs, pad_masks, _att), meta = self._call_embed_prefix(n_obs_steps=t, obs_history_is_pad=no_pad)

        state_slice = embs[:, meta["state_start"] : meta["state_end"]]
        state_mask = pad_masks[:, meta["state_start"] : meta["state_end"]]

        assert state_slice.shape == (1, t, meta["hidden_size"])
        assert state_mask.shape == (1, t)
        assert torch.all(state_mask), "all-valid: every state token should be unmasked"
        for ti in range(t):
            assert not torch.all(state_slice[:, ti] == 0), (
                f"state embedding at timestep {ti} should be non-zero"
            )

    # ------------------------------------------------------------------ #
    # Case 4: T>1, obs_history_is_pad=None → fallback to all-padded
    # ------------------------------------------------------------------ #
    def test_state_embedding_none_fallback(self):
        """obs_history_is_pad=None: state mask matches explicit all-padded."""
        t = 4
        (_, pm_none, _), meta = self._call_embed_prefix(n_obs_steps=t, obs_history_is_pad=None)

        pad = torch.ones(1, t, dtype=torch.bool)
        pad[:, -1] = False
        (_, pm_explicit, _), _ = self._call_embed_prefix(n_obs_steps=t, obs_history_is_pad=pad)

        state_none = pm_none[:, meta["state_start"] : meta["state_end"]]
        state_explicit = pm_explicit[:, meta["state_start"] : meta["state_end"]]

        assert torch.equal(state_none, state_explicit), (
            "None fallback should produce the same state mask as explicit all-padded"
        )

    # ------------------------------------------------------------------ #
    # Case 5: mixed batch — per-sample state masks are independent
    # ------------------------------------------------------------------ #
    def test_state_embedding_mixed_batch(self):
        """Heterogeneous pad masks: each sample's state mask is correct independently.

        Sample 0: [True, True, True, False] → only last unmasked
        Sample 1: [False, False, False, False] → all unmasked
        Sample 2: [True, False, False, False] → slot 0 masked, rest unmasked
        """
        t = 4
        pad = torch.tensor(
            [
                [True, True, True, False],
                [False, False, False, False],
                [True, False, False, False],
            ],
            dtype=torch.bool,
        )

        (embs, pad_masks, _att), meta = self._call_embed_prefix(
            n_obs_steps=t, obs_history_is_pad=pad, bsize=3
        )

        state_mask = pad_masks[:, meta["state_start"] : meta["state_end"]]
        assert state_mask.shape == (3, t)

        expected = torch.tensor(
            [
                [False, False, False, True],
                [True, True, True, True],
                [False, True, True, True],
            ],
            dtype=torch.bool,
        )
        assert torch.equal(state_mask, expected), (
            f"mixed batch state mask mismatch:\n  got:      {state_mask}\n  expected: {expected}"
        )

    # ------------------------------------------------------------------ #
    # forward / sample_actions produce state embeddings
    # ------------------------------------------------------------------ #
    def test_forward_path_invokes_state_proj(self):
        """Training path (T>1): state_proj is called once with (B, T, D).

        Drives embed_prefix directly (the common code path for both
        forward() and sample_actions()) and instruments state_proj to
        record the input shape.
        """
        model = self._make_mock_model(hidden_size=8)

        state_proj_calls = []
        orig_state_proj = model.state_proj

        def tracking_state_proj(x):
            state_proj_calls.append(x.shape)
            return orig_state_proj(x)

        model.state_proj = tracking_state_proj

        bsize, t, state_dim = 2, 4, 8
        prompt_len = 5
        model.video_encoder.num_frames = t

        pad = torch.ones(bsize, t, dtype=torch.bool)
        pad[:, -1] = False

        _legacy_embed_prefix(
            model,
            videos=[torch.zeros(bsize, t, 3, 8, 8)],
            vid_masks=[torch.ones(bsize, dtype=torch.bool)],
            lang_tokens=torch.zeros(bsize, prompt_len, dtype=torch.long),
            lang_masks=torch.ones(bsize, prompt_len, dtype=torch.bool),
            state=torch.randn(bsize, t, state_dim),
            response_tokens=torch.zeros(bsize, 4, dtype=torch.long),
            response_masks=torch.zeros(bsize, 4, dtype=torch.bool),
            metadata_tokens=torch.zeros(bsize, 3, dtype=torch.long),
            metadata_masks=torch.zeros(bsize, 3, dtype=torch.bool),
            obs_history_is_pad=pad,
        )

        assert len(state_proj_calls) == 1, "state_proj should be called exactly once"
        assert state_proj_calls[0] == (bsize, t, state_dim), (
            f"state_proj input shape should be (B, T, D), got {state_proj_calls[0]}"
        )

    def test_sample_actions_path_invokes_state_proj(self):
        """Inference path (T=1): state_proj is called once with (B, 1, D)."""
        model = self._make_mock_model(hidden_size=8)

        state_proj_calls = []
        orig_state_proj = model.state_proj

        def tracking_state_proj(x):
            state_proj_calls.append(x.shape)
            return orig_state_proj(x)

        model.state_proj = tracking_state_proj

        bsize, state_dim = 1, 8
        prompt_len = 5

        _legacy_embed_prefix(
            model,
            videos=[torch.zeros(bsize, 1, 3, 8, 8)],
            vid_masks=[torch.ones(bsize, dtype=torch.bool)],
            lang_tokens=torch.zeros(bsize, prompt_len, dtype=torch.long),
            lang_masks=torch.ones(bsize, prompt_len, dtype=torch.bool),
            state=torch.randn(bsize, 1, state_dim),
            response_tokens=torch.zeros(bsize, 4, dtype=torch.long),
            response_masks=torch.zeros(bsize, 4, dtype=torch.bool),
            metadata_tokens=torch.zeros(bsize, 3, dtype=torch.long),
            metadata_masks=torch.zeros(bsize, 3, dtype=torch.bool),
            obs_history_is_pad=None,
        )

        assert len(state_proj_calls) == 1, "state_proj should be called exactly once"
        assert state_proj_calls[0] == (bsize, 1, state_dim), (
            f"state_proj input shape should be (B, 1, D), got {state_proj_calls[0]}"
        )


class TestPI07EmbedPrefixInvariants:
    """CPU-only tests for layout invariants enforced by ``embed_prefix``.

    The dispatcher in ``embed_prefix`` requires that any items flagged
    ``exclude_from_cross_attention=True`` form a contiguous trailing
    run, since ``num_cross_att_tokens`` is the prefix length up to (but
    not including) the first excluded item. If a non-excluded item ever
    appeared after an excluded one, the count would silently undercount
    the cross-attention scope. The guard raises ``ValueError`` so this
    test pins the behavior — without it, the invariant is one
    accidental refactor away from being silently bypassed.
    """

    def _make_mock_model(self, hidden_size: int = 8):
        """Reuse the lightweight CPU stub from
        :class:`TestPI07PaligemmaLowLevelStateEmbedding` so this test
        class has no PaliGemma / GPU dependency.
        """
        return TestPI07PaligemmaLowLevelStateEmbedding._make_mock_model(hidden_size=hidden_size)

    @staticmethod
    def _text_item(bsize: int, length: int, *, exclude: bool) -> ContextItem:
        """Minimal ``text`` ``ContextItem`` for invariant testing — the
        mock ``embed_language_tokens`` returns zeros, so token IDs and
        masks just need to have the right shape.
        """
        return ContextItem(
            data=torch.zeros(bsize, length, dtype=torch.long),
            item_type="text",
            pad_mask=torch.ones(bsize, length, dtype=torch.bool),
            attention="continue",
            exclude_from_cross_attention=exclude,
        )

    def test_excluded_followed_by_included_raises(self):
        """A non-excluded item after an excluded one is illegal."""
        model = self._make_mock_model()
        items = [
            self._text_item(1, 3, exclude=False),
            self._text_item(1, 2, exclude=True),
            self._text_item(1, 1, exclude=False),
        ]
        with pytest.raises(ValueError, match="contiguous trailing run"):
            PI07PaligemmaLowLevelFlowMatching.embed_prefix(model, items)

    def test_all_included_then_all_excluded_ok(self):
        """The legal layout (excluded items as a trailing run) returns
        a ``num_cross_att_tokens`` equal to the summed lengths of the
        non-excluded prefix."""
        model = self._make_mock_model()
        items = [
            self._text_item(1, 3, exclude=False),
            self._text_item(1, 2, exclude=False),
            self._text_item(1, 4, exclude=True),
            self._text_item(1, 1, exclude=True),
        ]
        _embs, _pad, _att, num_cross = PI07PaligemmaLowLevelFlowMatching.embed_prefix(model, items)
        assert num_cross == 5  # 3 + 2; the trailing 4 + 1 are excluded.

    def test_train_inference_prefix_item_order_parity(self):
        """Training (``include_discrete_actions=True``) and inference (``False``)
        build the same prefix block ordering, differing only by the trailing
        training-only ``Action:`` + discrete-action items. Both paths share
        ``_build_prefix_items``, so a one-sided block append regresses loudly here.
        """
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=8)
        bsz = 2
        common = {
            "videos": [torch.zeros(bsz, 1, 3, 8, 8)],
            "vid_masks": [torch.ones(bsz, dtype=torch.bool)],
            "lang_tokens": torch.zeros(bsz, 5, dtype=torch.long),
            "lang_masks": torch.ones(bsz, 5, dtype=torch.bool),
            "state": torch.zeros(bsz, 1, 8),
            "response_tokens": torch.zeros(bsz, RESPONSE_MAX_LENGTH, dtype=torch.long),
            "response_masks": torch.zeros(bsz, RESPONSE_MAX_LENGTH, dtype=torch.bool),
            "metadata_tokens": torch.zeros(bsz, METADATA_MAX_LENGTH, dtype=torch.long),
            "metadata_masks": torch.zeros(bsz, METADATA_MAX_LENGTH, dtype=torch.bool),
            "subgoal_videos": [torch.zeros(bsz, 3, 224, 224)],
            "subgoal_vid_masks": [torch.zeros(bsz, dtype=torch.bool)],
        }

        infer_items = _legacy_embed_prefix(model, return_items=True, **common)
        train_items = _legacy_embed_prefix(
            model,
            return_items=True,
            discrete_actions=torch.zeros(bsz, DISCRETE_ACTION_MAX_LENGTH, dtype=torch.long),
            discrete_action_masks=torch.ones(bsz, DISCRETE_ACTION_MAX_LENGTH, dtype=torch.bool),
            **common,
        )

        infer_types = [it.item_type for it in infer_items]
        train_types = [it.item_type for it in train_items]

        # Inference prefix is a strict prefix of the training prefix.
        assert train_types[: len(infer_types)] == infer_types, (
            f"train/inference prefix order diverged:\n  infer={infer_types}\n  train={train_types}"
        )
        # Training appends exactly the "Action: " indicator (text) + discrete actions.
        assert train_types[len(infer_types) :] == ["text", "discrete_action"]
        # Subgoal is embedded as an "image" block (A3) and sits before the
        # training-only discrete-action tail.
        assert "image" in infer_types
        assert train_types.index("discrete_action") > max(
            i for i, t in enumerate(train_types) if t == "image"
        )


class TestPI07PaligemmaLowLevelResponseEmbedding:
    """CPU-only tests verifying response token masking in ``embed_prefix``.

    ``response_tokens`` / ``response_masks`` come from
    :meth:`PI07PaligemmaLowLevelPolicy.prepare_response` (real PaliGemma tokenizer),
    then are fed into a lightweight ``embed_prefix`` mock.

    Prefix layout (no-optionals simplified):
        [video] [lang] ["State: "] [state(T)] [", "] [response] [", "] ...

    The ``", "`` separator before the response block is masked/unmasked
    based on ``sample_has_response = response_masks.any(dim=1)``.
    """

    _STATE_LEAD_IDS = [10, 11, 12]  # "State: "
    _COMMA_IDS = [20]  # ", "
    _COLON_NL_IDS = [30]  # ":\n"
    _SUBGOAL_LEAD_IDS = [40, 41]  # "Subgoal: "
    _ACTION_LEAD_IDS = [50, 51]  # "Action: "

    _cached_prepare_policy: object | None = None

    @classmethod
    def _get_prepare_policy(cls) -> object:
        if cls._cached_prepare_policy is None:
            policy = object.__new__(PI07PaligemmaLowLevelPolicy)
            policy.config = TestPI07PaligemmaLowLevelIntegration._make_config()
            policy.language_tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
            cls._cached_prepare_policy = policy
        return cls._cached_prepare_policy

    @classmethod
    def _prepare_response(cls, responses: list[str], *, omit_response_key: bool = False):
        """Build ``(response_tokens, response_masks)`` via ``prepare_response``."""
        bsz = len(responses)
        device = torch.device("cpu")
        batch: dict = {"state": torch.zeros(bsz, 1, MAX_STATE_DIM, device=device)}
        if not omit_response_key:
            batch["response"] = responses

        policy = cls._get_prepare_policy()
        return PI07PaligemmaLowLevelPolicy.prepare_response(policy, batch)

    @classmethod
    def _fake_tokenizer(cls):
        ids_by_str = {
            "State: ": cls._STATE_LEAD_IDS,
            ", ": cls._COMMA_IDS,
            ":\n": cls._COLON_NL_IDS,
            "Subgoal: ": cls._SUBGOAL_LEAD_IDS,
            "Action: ": cls._ACTION_LEAD_IDS,
        }

        class _FakeTokenizer:
            @staticmethod
            def encode(s, add_special_tokens=False):
                assert add_special_tokens is False
                return ids_by_str[s]

        return _FakeTokenizer()

    @classmethod
    def _make_mock_model(cls, hidden_size: int = 8):
        h = hidden_size
        model = object.__new__(PI07PaligemmaLowLevelFlowMatching)

        # num_image_tokens mirrors embed_image's 4-token output below, so the
        # image-tower skip path (zeros of that width) matches the run path.
        _text_cfg = type("_TextConfig", (), {"hidden_size": h, "num_image_tokens": 4})()
        _pg_cfg = type("_PaliGemmaConfig", (), {"text_config": _text_cfg})()
        _stub_cfg = type("_StubConfig", (), {"paligemma_config": _pg_cfg})()

        class _PaliGemmaStub:
            config = _stub_cfg

            @staticmethod
            def embed_language_tokens(tokens):
                return torch.zeros(tokens.shape[0], tokens.shape[1], h, dtype=torch.bfloat16)

            @staticmethod
            def embed_image(img):
                return torch.zeros(img.shape[0], 4, h, dtype=torch.bfloat16)

            @staticmethod
            def embed_discrete_actions(tokens):
                return torch.zeros(tokens.shape[0], tokens.shape[1], h, dtype=torch.bfloat16)

        _video_enc = type(
            "_VideoEncoderStub",
            (),
            {"num_video_tokens": 6, "num_frames": 1},
        )()

        model.paligemma_with_expert = _PaliGemmaStub()
        model.video_encoder = _video_enc
        model.language_tokenizer = cls._fake_tokenizer()
        model.state_proj = lambda x: torch.zeros(x.shape[0], x.shape[1], h, dtype=torch.bfloat16)

        def _embed_video(video, obs_history_is_pad=None):
            t = video.shape[1]
            max_t = model.video_encoder.num_frames
            if not 1 <= t <= max_t:
                raise ValueError(f"Expected 1 <= T <= {max_t}; got {t}.")
            return torch.zeros(video.shape[0], 6, h, dtype=torch.bfloat16)

        model.embed_video = _embed_video
        return model

    @classmethod
    def _call_embed_prefix(
        cls,
        *,
        bsize: int,
        response_tokens: torch.Tensor,
        response_masks: torch.Tensor,
        n_obs_steps: int = 1,
        prompt_len: int = 5,
        metadata_len: int = 3,
        hidden_size: int = 8,
    ):
        """Drive embed_prefix and return (embs, pad_masks, att_masks) plus
        a metadata dict with index boundaries for the comma + response slice."""
        model = cls._make_mock_model(hidden_size=hidden_size)
        model.video_encoder.num_frames = n_obs_steps

        (embs, pad_masks, att_masks) = _legacy_embed_prefix(
            model,
            videos=[torch.zeros(bsize, n_obs_steps, 3, 8, 8)],
            vid_masks=[torch.ones(bsize, dtype=torch.bool)],
            lang_tokens=torch.zeros(bsize, prompt_len, dtype=torch.long),
            lang_masks=torch.ones(bsize, prompt_len, dtype=torch.bool),
            state=torch.zeros(bsize, n_obs_steps, 8),
            response_tokens=response_tokens,
            response_masks=response_masks,
            metadata_tokens=torch.zeros(bsize, metadata_len, dtype=torch.long),
            metadata_masks=torch.zeros(bsize, metadata_len, dtype=torch.bool),
        )

        vid_tokens = 6
        state_lead = len(cls._STATE_LEAD_IDS)
        comma_len = len(cls._COMMA_IDS)
        resp_len = response_tokens.shape[1]

        comma_start = vid_tokens + prompt_len + state_lead + n_obs_steps
        comma_end = comma_start + comma_len
        resp_start = comma_end
        resp_end = resp_start + resp_len

        return (embs, pad_masks, att_masks), {
            "comma_start": comma_start,
            "comma_end": comma_end,
            "resp_start": resp_start,
            "resp_end": resp_end,
        }

    # ------------------------------------------------------------------ #
    # Test 1: response="" — all response tokens padded, ", " padded
    # ------------------------------------------------------------------ #
    def test_empty_response_comma_and_tokens_padded(self):
        """Empty response: ``prepare_response`` yields all-pad masks; the \", \"
        separator and all response slots in ``embed_prefix`` must be padded."""
        response_tokens, response_masks = self._prepare_response([""])
        assert response_tokens.shape[0] == 1
        assert response_tokens.shape[1] == RESPONSE_MAX_LENGTH
        assert not response_masks.any(), "prepare_response('') should produce no real tokens"

        bsize = response_tokens.shape[0]
        (_, pad_masks, _), meta = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=response_tokens,
            response_masks=response_masks,
        )

        comma_mask = pad_masks[:, meta["comma_start"] : meta["comma_end"]]
        assert torch.all(~comma_mask), 'empty response: ", " separator should be fully padded (masked out)'

        resp_mask = pad_masks[:, meta["resp_start"] : meta["resp_end"]]
        assert torch.all(~resp_mask), "empty response: all response tokens should be padded"

    # ------------------------------------------------------------------ #
    # Test 2: response!="" — ", " and response tokens unmasked
    # ------------------------------------------------------------------ #
    def test_nonempty_response_comma_and_tokens_unmasked(self):
        """Non-empty response: ``prepare_response`` marks real tokens; the \", \"
        separator must be unmasked and response pad_mask must match ``response_masks``."""
        text = "Grasp the red block and place it in the bin."
        response_tokens, response_masks = self._prepare_response([text])
        assert response_masks.any(), "prepare_response should mark at least one real token"
        assert response_tokens.shape[1] == RESPONSE_MAX_LENGTH

        bsize = response_tokens.shape[0]
        (_, pad_masks, _), meta = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=response_tokens,
            response_masks=response_masks,
        )

        comma_mask = pad_masks[:, meta["comma_start"] : meta["comma_end"]]
        assert torch.all(comma_mask), 'non-empty response: ", " separator should be unmasked'

        resp_mask = pad_masks[:, meta["resp_start"] : meta["resp_end"]]
        assert torch.equal(resp_mask, response_masks), (
            "response pad_mask should match prepare_response masks exactly"
        )

    # ------------------------------------------------------------------ #
    # Test 3: mixed batch — sample 0 empty, sample 1 has response
    # ------------------------------------------------------------------ #
    def test_mixed_batch_response_masking(self):
        """Mixed batch: ``prepare_response`` on ``[\"\", real]``; per-sample comma
        and response pad_masks must follow ``response_masks.any(dim=1)``."""
        response_tokens, response_masks = self._prepare_response(
            ["", "Grasp the red block and place it in the bin."]
        )
        assert not response_masks[0].any()
        assert response_masks[1].any()

        bsize = response_tokens.shape[0]
        (_, pad_masks, _), meta = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=response_tokens,
            response_masks=response_masks,
        )

        comma_mask = pad_masks[:, meta["comma_start"] : meta["comma_end"]]
        assert torch.all(~comma_mask[0]), 'sample 0 (empty response): ", " should be padded'
        assert torch.all(comma_mask[1]), 'sample 1 (real response): ", " should be unmasked'

        resp_mask = pad_masks[:, meta["resp_start"] : meta["resp_end"]]
        assert torch.equal(resp_mask, response_masks)
        assert torch.all(~resp_mask[0])

    # ------------------------------------------------------------------ #
    # Test 4: forward vs sample_actions parity for response masking
    # ------------------------------------------------------------------ #
    def test_response_masking_forward_vs_sample_actions_parity(self):
        """Identical ``prepare_response`` outputs produce identical ``embed_prefix``
        pad_masks (parity for empty vs non-empty). Missing ``response`` key matches
        explicit ``[\"\"]`` (same as ``_hydrate_optional_conditioning_batch``)."""
        empty_tokens_a, empty_masks_a = self._prepare_response([""])
        empty_tokens_b, empty_masks_b = self._prepare_response([""])
        assert torch.equal(empty_tokens_a, empty_tokens_b)
        assert torch.equal(empty_masks_a, empty_masks_b)

        empty_tokens_c, empty_masks_c = self._prepare_response([""], omit_response_key=True)
        assert torch.equal(empty_tokens_a, empty_tokens_c)
        assert torch.equal(empty_masks_a, empty_masks_c)

        text = "Open the drawer slowly."
        real_tokens_a, real_masks_a = self._prepare_response([text])
        real_tokens_b, real_masks_b = self._prepare_response([text])
        assert torch.equal(real_tokens_a, real_tokens_b)
        assert torch.equal(real_masks_a, real_masks_b)

        bsize = 1
        (_, pm_empty_1, _), meta = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=empty_tokens_a,
            response_masks=empty_masks_a,
        )
        (_, pm_empty_2, _), _ = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=empty_tokens_b,
            response_masks=empty_masks_b,
        )
        assert torch.equal(pm_empty_1, pm_empty_2), (
            "two prepare_response(empty) runs should yield identical embed_prefix pad_masks"
        )

        (_, pm_real_1, _), _ = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=real_tokens_a,
            response_masks=real_masks_a,
        )
        (_, pm_real_2, _), _ = self._call_embed_prefix(
            bsize=bsize,
            response_tokens=real_tokens_b,
            response_masks=real_masks_b,
        )
        assert torch.equal(pm_real_1, pm_real_2), (
            "two prepare_response(non-empty) runs should yield identical embed_prefix pad_masks"
        )

        comma_empty = pm_empty_1[:, meta["comma_start"] : meta["comma_end"]]
        comma_real = pm_real_1[:, meta["comma_start"] : meta["comma_end"]]
        assert not torch.equal(comma_empty, comma_real), (
            'empty vs non-empty response should differ in the ", " separator mask'
        )

        resp_empty = pm_empty_1[:, meta["resp_start"] : meta["resp_end"]]
        resp_real = pm_real_1[:, meta["resp_start"] : meta["resp_end"]]
        assert not torch.equal(resp_empty, resp_real), (
            "empty vs non-empty response should differ in response token masks"
        )


class TestPI07PaligemmaLowLevelMetadataEmbedding:
    """CPU-only tests for metadata pad masks in ``embed_prefix``.

    ``metadata_tokens`` / ``metadata_masks`` come from
    :meth:`PI07PaligemmaLowLevelPolicy.prepare_metadata` (real tokenizer + the
    same ``speed`` / ``quality`` / ``mistake`` + ``*_is_pad`` rules as training).

    Prefix slice after the response block (subgoal now sits at the tail):
        ... [response] [\", \" md] [metadata] [\", \" sg] [\"Subgoal: \"] [\":\\n\"]

    The metadata ``\", \"`` separator uses ``sample_has_metadata = metadata_masks.any(dim=1)``.
    """

    _STATE_LEAD_LEN = len(TestPI07PaligemmaLowLevelResponseEmbedding._STATE_LEAD_IDS)
    _COMMA_LEN = len(TestPI07PaligemmaLowLevelResponseEmbedding._COMMA_IDS)
    _MD_COMMA_LEN = len(TestPI07PaligemmaLowLevelResponseEmbedding._COMMA_IDS)
    _VID_TOKENS = 6

    @classmethod
    def _prepare_metadata(cls, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        policy = TestPI07PaligemmaLowLevelResponseEmbedding._get_prepare_policy()
        return PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

    @classmethod
    def _metadata_batch(
        cls,
        bsz: int,
        *,
        speed_pad: torch.Tensor | bool,
        quality_pad: torch.Tensor | bool,
        mistake_pad: torch.Tensor | bool,
        speed_val: float = 50.0,
        quality_val: float = 3.0,
        mistake_val: bool = False,
    ) -> dict:
        device = torch.device("cpu")
        batch = {
            "state": torch.zeros(bsz, 1, MAX_STATE_DIM, device=device),
            "speed": torch.full((bsz,), speed_val, dtype=torch.float32, device=device),
            "quality": torch.full((bsz,), quality_val, dtype=torch.float32, device=device),
            "mistake": torch.full((bsz,), mistake_val, dtype=torch.bool, device=device),
        }
        for key, pad in (
            ("speed_is_pad", speed_pad),
            ("quality_is_pad", quality_pad),
            ("mistake_is_pad", mistake_pad),
        ):
            t = torch.as_tensor(pad, dtype=torch.bool, device=device).reshape(-1)
            if t.numel() == 1:
                t = t.expand(bsz)
            batch[key] = t
        return batch

    @classmethod
    def _call_embed_prefix_with_metadata(
        cls,
        *,
        bsize: int,
        response_tokens: torch.Tensor,
        response_masks: torch.Tensor,
        metadata_tokens: torch.Tensor,
        metadata_masks: torch.Tensor,
        n_obs_steps: int = 1,
        prompt_len: int = 5,
        hidden_size: int = 8,
    ) -> tuple[torch.Tensor, dict]:
        """Run ``embed_prefix`` with real-shaped metadata; return ``pad_masks`` and
        slice indices for the metadata ``\", \"`` and metadata token block."""
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=hidden_size)
        assert metadata_tokens.shape[1] == METADATA_MAX_LENGTH

        (_, pad_masks, _) = _legacy_embed_prefix(
            model,
            videos=[torch.zeros(bsize, n_obs_steps, 3, 8, 8)],
            vid_masks=[torch.ones(bsize, dtype=torch.bool)],
            lang_tokens=torch.zeros(bsize, prompt_len, dtype=torch.long),
            lang_masks=torch.ones(bsize, prompt_len, dtype=torch.bool),
            state=torch.zeros(bsize, n_obs_steps, 8),
            response_tokens=response_tokens,
            response_masks=response_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            subgoal_videos=(),
            subgoal_vid_masks=(),
        )

        resp_len = response_tokens.shape[1]
        # Response comma (before response tokens)
        resp_comma_start = cls._VID_TOKENS + prompt_len + cls._STATE_LEAD_LEN + n_obs_steps
        resp_comma_end = resp_comma_start + cls._COMMA_LEN
        resp_start = resp_comma_end
        resp_end = resp_start + resp_len
        # Metadata block now sits directly after the response (subgoal moved to
        # the tail, so no subgoal header precedes the metadata).
        md_comma_start = resp_end
        md_comma_end = md_comma_start + cls._MD_COMMA_LEN
        meta_start = md_comma_end
        meta_end = meta_start + METADATA_MAX_LENGTH

        return pad_masks, {
            "md_comma_start": md_comma_start,
            "md_comma_end": md_comma_end,
            "meta_start": meta_start,
            "meta_end": meta_end,
        }

    @classmethod
    def _empty_response(cls, bsz: int) -> tuple[torch.Tensor, torch.Tensor]:
        return TestPI07PaligemmaLowLevelResponseEmbedding._prepare_response([""] * bsz)

    # ------------------------------------------------------------------ #
    # All metadata dropped (all *_is_pad True) — same as inference defaults
    # ------------------------------------------------------------------ #
    def test_all_metadata_dropped_md_comma_and_tokens_padded(self):
        """All three fields treated as pad → empty metadata string; the metadata
        ``, `` separator and all metadata slots must be padded in ``embed_prefix``."""
        bsz = 1
        batch = self._metadata_batch(
            bsz,
            speed_pad=True,
            quality_pad=True,
            mistake_pad=True,
        )
        meta_tokens, meta_masks = self._prepare_metadata(batch)
        assert meta_tokens.shape == (bsz, METADATA_MAX_LENGTH)
        assert not meta_masks.any()

        rt, rm = self._empty_response(bsz)
        pad_masks, sl = self._call_embed_prefix_with_metadata(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=meta_tokens,
            metadata_masks=meta_masks,
        )

        md_comma = pad_masks[:, sl["md_comma_start"] : sl["md_comma_end"]]
        assert torch.all(~md_comma), 'all-metadata-pad: metadata ", " should be fully padded'

        meta_pad = pad_masks[:, sl["meta_start"] : sl["meta_end"]]
        assert torch.all(~meta_pad), "all-metadata-pad: every metadata token should be padded"

    # ------------------------------------------------------------------ #
    # All metadata present
    # ------------------------------------------------------------------ #
    def test_all_metadata_present_md_comma_unmasked(self):
        """All fields real → ``prepare_metadata`` marks tokens; metadata ``, ``
        unmasked and prefix slice equals ``metadata_masks``."""
        bsz = 1
        batch = self._metadata_batch(
            bsz,
            speed_pad=False,
            quality_pad=False,
            mistake_pad=False,
        )
        meta_tokens, meta_masks = self._prepare_metadata(batch)
        assert meta_masks.any()

        rt, rm = self._empty_response(bsz)
        pad_masks, sl = self._call_embed_prefix_with_metadata(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=meta_tokens,
            metadata_masks=meta_masks,
        )

        md_comma = pad_masks[:, sl["md_comma_start"] : sl["md_comma_end"]]
        assert torch.all(md_comma), 'all-metadata-real: metadata ", " should be unmasked'

        meta_pad = pad_masks[:, sl["meta_start"] : sl["meta_end"]]
        assert torch.equal(meta_pad, meta_masks)

    # ------------------------------------------------------------------ #
    # Exhaustive single-sample pad combinations (speed / quality / mistake)
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        "spad,qpad,mpad",
        [
            (True, True, True),
            (False, False, False),
            (False, True, True),
            (True, False, True),
            (True, True, False),
            (False, False, True),
            (False, True, False),
            (True, False, False),
        ],
    )
    def test_metadata_is_pad_exhaustive_single_sample(self, spad, qpad, mpad):
        """Every combination of ``*_is_pad`` (True = field dropped) must drive
        ``sample_has_metadata`` consistently in ``embed_prefix``."""
        bsz = 1
        batch = self._metadata_batch(bsz, speed_pad=spad, quality_pad=qpad, mistake_pad=mpad)
        meta_tokens, meta_masks = self._prepare_metadata(batch)
        expect_any_meta = not (spad and qpad and mpad)
        assert meta_masks.any().item() == expect_any_meta

        rt, rm = self._empty_response(bsz)
        pad_masks, sl = self._call_embed_prefix_with_metadata(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=meta_tokens,
            metadata_masks=meta_masks,
        )

        sample_has = meta_masks.any(dim=1)
        md_comma = pad_masks[:, sl["md_comma_start"] : sl["md_comma_end"]]
        assert torch.equal(md_comma, sample_has[:, None].expand_as(md_comma)), (
            "metadata comma mask must follow metadata_masks.any(dim=1)"
        )

        meta_slice = pad_masks[:, sl["meta_start"] : sl["meta_end"]]
        assert torch.equal(meta_slice, meta_masks)

    # ------------------------------------------------------------------ #
    # Mixed batch: heterogeneous per-sample metadata presence
    # ------------------------------------------------------------------ #
    def test_mixed_batch_metadata_masking(self):
        """Per-sample ``*_is_pad`` patterns must not leak across the batch."""
        bsz = 3
        batch = {
            "state": torch.zeros(bsz, 1, MAX_STATE_DIM),
            "speed": torch.tensor([10.0, 50.0, 90.0]),
            "quality": torch.tensor([1.0, 2.0, 3.0]),
            "mistake": torch.tensor([False, True, False]),
            # Sample 0: all pad. Sample 1: all real. Sample 2: only speed real.
            "speed_is_pad": torch.tensor([True, False, False]),
            "quality_is_pad": torch.tensor([True, False, True]),
            "mistake_is_pad": torch.tensor([True, False, True]),
        }
        meta_tokens, meta_masks = self._prepare_metadata(batch)
        assert not meta_masks[0].any()
        assert meta_masks[1].any()
        assert meta_masks[2].any()

        rt, rm = self._empty_response(bsz)
        pad_masks, sl = self._call_embed_prefix_with_metadata(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=meta_tokens,
            metadata_masks=meta_masks,
        )

        md_comma = pad_masks[:, sl["md_comma_start"] : sl["md_comma_end"]]
        assert torch.all(~md_comma[0])
        assert torch.all(md_comma[1])
        assert torch.all(md_comma[2])

        meta_slice = pad_masks[:, sl["meta_start"] : sl["meta_end"]]
        assert torch.equal(meta_slice, meta_masks)

    # ------------------------------------------------------------------ #
    # Missing keys (inference) vs explicit all-pad
    # ------------------------------------------------------------------ #
    def test_metadata_missing_keys_matches_explicit_all_pad(self):
        """Batch with only ``state`` (``_hydrate_metadata_batch`` defaults) must
        match an explicit batch with all ``*_is_pad`` True."""
        bsz = 1
        batch_missing = {"state": torch.zeros(bsz, 1, MAX_STATE_DIM)}
        t_miss, m_miss = self._prepare_metadata(batch_missing)

        batch_explicit = self._metadata_batch(
            bsz,
            speed_pad=True,
            quality_pad=True,
            mistake_pad=True,
        )
        t_exp, m_exp = self._prepare_metadata(batch_explicit)

        assert torch.equal(t_miss, t_exp)
        assert torch.equal(m_miss, m_exp)

        rt, rm = self._empty_response(bsz)
        pm_a, sl = self._call_embed_prefix_with_metadata(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=t_miss,
            metadata_masks=m_miss,
        )
        pm_b, _ = self._call_embed_prefix_with_metadata(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=t_exp,
            metadata_masks=m_exp,
        )
        assert torch.equal(pm_a, pm_b)


def _find_free_port() -> int:
    """Pick an ephemeral localhost port for a gloo rendezvous."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ddp_image_skip_worker(rank: int, world_size: int, port: int, scenario: str) -> None:
    """Subprocess body for ``test_ddp_lockstep_image_tower_skip``.

    Each rank calls ``_global_run_image_tower`` — the single, hoisted collective
    — on a rank-specific prefix and asserts the decision follows the *global* OR
    (not the rank-local ``.any()``), and that a cross-rank presence mismatch
    raises rather than silently desyncing. Must be module-level so the macOS
    ``spawn`` start method can pickle it. Asserts raise in-process;
    ``mp.spawn(join=True)`` re-raises them in the parent test.
    """
    import os

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        bsz, h = 2, 8
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=h)

        def _image_item(mask: torch.Tensor) -> ContextItem:
            return ContextItem(
                data=torch.full((bsz, 3, 224, 224), 0.3),
                item_type="image",
                pad_mask=mask,
                attention="continue",
            )

        # A non-image filler so every rank's prefix is non-empty (as in production,
        # where video/text/state items always precede the subgoal block).
        text_item = ContextItem(
            data=torch.zeros(bsz, 3, dtype=torch.long),
            item_type="text",
            pad_mask=torch.ones(bsz, 3, dtype=torch.bool),
            attention="continue",
        )

        if scenario == "divergent":
            # rank 0 has no real subgoal, rank 1 does → global OR True, so BOTH
            # ranks must run (rank 0 despite its all-False local mask). This is
            # the lockstep guarantee the hoisted OR-reduce provides.
            mask = torch.zeros(bsz, dtype=torch.bool) if rank == 0 else torch.tensor([False, True])
            assert model._global_run_image_tower([text_item, _image_item(mask)]), (
                f"rank {rank}/divergent: expected global RUN despite local mask {mask.tolist()}"
            )
        elif scenario == "all_padded":
            # No rank has a real subgoal → global OR False → both skip.
            items = [text_item, _image_item(torch.zeros(bsz, dtype=torch.bool))]
            assert not model._global_run_image_tower(items), f"rank {rank}/all_padded: expected SKIP"
        elif scenario == "presence_divergence":
            # rank 0's prefix has an image item, rank 1's has none. The helper's
            # presence-divergence check must fire (loud RuntimeError) rather than
            # let the per-rank collective count silently desync — the exact
            # failure the hoist restores protection against (CLAUDE.md rule 5).
            items = [text_item, _image_item(torch.tensor([False, True]))] if rank == 0 else [text_item]
            try:
                model._global_run_image_tower(items)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"rank {rank}/presence_divergence: expected RuntimeError, got none")
        else:
            raise ValueError(f"unknown scenario {scenario!r}")
    finally:
        torch.distributed.destroy_process_group()


class TestPI07PaligemmaLowLevelSubgoalEmbedding:
    """CPU-only tests for subgoal block pad masks in ``embed_prefix``.

    ``subgoal_videos`` / ``subgoal_vid_masks`` come from
    :meth:`PI07PaligemmaLowLevelPolicy.prepare_subgoal_images` (same
    ``image_features`` / ``subgoal{k}`` layout as training).

    Subgoal sits at the prefix tail (after the metadata block): ``\", \"``
    (subgoal) → ``\"Subgoal: \"`` → image tokens per camera (via
    ``embed_image``).  ``sample_has_subgoal`` is
    ``torch.stack(subgoal_img_masks, dim=0).any(dim=0)``.
    """

    _VID = 6
    _STATE_LEAD = len(TestPI07PaligemmaLowLevelResponseEmbedding._STATE_LEAD_IDS)
    _COMMA = len(TestPI07PaligemmaLowLevelResponseEmbedding._COMMA_IDS)
    _SG_START = len(TestPI07PaligemmaLowLevelResponseEmbedding._SUBGOAL_LEAD_IDS)
    _N_SG_TOKENS = 4  # mock ``paligemma_with_expert.embed_image`` token count

    @classmethod
    def _prepare_subgoal(cls, batch: dict) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        policy = TestPI07PaligemmaLowLevelResponseEmbedding._get_prepare_policy()
        return PI07PaligemmaLowLevelPolicy.prepare_subgoal_images(policy, batch)

    @classmethod
    def _empty_response_metadata(cls, bsz: int) -> tuple[torch.Tensor, ...]:
        rt, rm = TestPI07PaligemmaLowLevelResponseEmbedding._prepare_response([""] * bsz)
        mb = TestPI07PaligemmaLowLevelMetadataEmbedding._metadata_batch(
            bsz,
            speed_pad=True,
            quality_pad=True,
            mistake_pad=True,
        )
        mt, mm = TestPI07PaligemmaLowLevelMetadataEmbedding._prepare_metadata(mb)
        return rt, rm, mt, mm

    @classmethod
    def _subgoal_batch(
        cls,
        bsz: int,
        *,
        include_tensors: bool,
        subgoal_is_pad: torch.Tensor | bool,
        fill: float = 0.42,
    ) -> dict:
        """Build a batch dict for ``prepare_subgoal_images``."""
        device = torch.device("cpu")
        batch: dict = {"state": torch.zeros(bsz, 1, MAX_STATE_DIM, device=device)}
        pad = torch.as_tensor(subgoal_is_pad, dtype=torch.bool, device=device).reshape(-1)
        if pad.numel() == 1:
            pad = pad.expand(bsz)
        batch["subgoal_is_pad"] = pad
        if include_tensors:
            img = torch.full((bsz, 3, 224, 224), fill, device=device)
            batch["subgoal0"] = img
            batch["subgoal1"] = img * 0.9
        return batch

    @classmethod
    def _call_embed_prefix_with_subgoals(
        cls,
        *,
        bsize: int,
        response_tokens: torch.Tensor,
        response_masks: torch.Tensor,
        metadata_tokens: torch.Tensor,
        metadata_masks: torch.Tensor,
        subgoal_videos: list[torch.Tensor],
        subgoal_vid_masks: list[torch.Tensor],
        n_obs_steps: int = 1,
        prompt_len: int = 5,
        hidden_size: int = 8,
    ) -> tuple[torch.Tensor, dict]:
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=hidden_size)
        model.video_encoder.num_frames = n_obs_steps
        assert len(subgoal_videos) == len(subgoal_vid_masks)

        (_, pad_masks, _) = _legacy_embed_prefix(
            model,
            videos=[torch.zeros(bsize, n_obs_steps, 3, 8, 8)],
            vid_masks=[torch.ones(bsize, dtype=torch.bool)],
            lang_tokens=torch.zeros(bsize, prompt_len, dtype=torch.long),
            lang_masks=torch.ones(bsize, prompt_len, dtype=torch.bool),
            state=torch.zeros(bsize, n_obs_steps, 8),
            response_tokens=response_tokens,
            response_masks=response_masks,
            metadata_tokens=metadata_tokens,
            metadata_masks=metadata_masks,
            subgoal_videos=subgoal_videos,
            subgoal_vid_masks=subgoal_vid_masks,
        )

        resp_len = response_tokens.shape[1]
        rcomma_lo = cls._VID + prompt_len + cls._STATE_LEAD + n_obs_steps
        rcomma_hi = rcomma_lo + cls._COMMA
        resp_lo = rcomma_hi
        resp_hi = resp_lo + resp_len
        # Metadata block (md comma + METADATA_MAX_LENGTH tokens) now precedes the
        # subgoal block, so the subgoal comma starts after it.
        sg_comma_lo = resp_hi + cls._COMMA + METADATA_MAX_LENGTH
        sg_comma_hi = sg_comma_lo + cls._COMMA
        sg_start_lo = sg_comma_hi
        sg_start_hi = sg_start_lo + cls._SG_START
        sg_vid_lo = sg_start_hi
        n_cam = len(subgoal_vid_masks)
        sg_vid_hi = sg_vid_lo + n_cam * cls._N_SG_TOKENS

        sample_has_sg = torch.stack(subgoal_vid_masks, dim=0).any(dim=0)

        return pad_masks, {
            "sg_comma_lo": sg_comma_lo,
            "sg_comma_hi": sg_comma_hi,
            "sg_start_lo": sg_start_lo,
            "sg_start_hi": sg_start_hi,
            "sg_vid_lo": sg_vid_lo,
            "sg_vid_hi": sg_vid_hi,
            "n_cam": n_cam,
            "sample_has_subgoal": sample_has_sg,
        }

    # ------------------------------------------------------------------ #
    # No subgoal tensors (inference-style) — all subgoal slots padded
    # ------------------------------------------------------------------ #
    def test_no_subgoal_keys_sg_block_fully_padded(self):
        """Batch with only ``state`` → fabricated zero slots and all-false masks;
        ``sample_has_subgoal`` is false → sg comma, header, and vision tokens padded."""
        bsz = 1
        batch = {"state": torch.zeros(bsz, 1, MAX_STATE_DIM)}
        sg_videos, sg_masks = self._prepare_subgoal(batch)
        assert len(sg_videos) == 2 and len(sg_masks) == 2
        assert all(not m.any() for m in sg_masks)

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pad_masks, sl = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=sg_videos,
            subgoal_vid_masks=sg_masks,
        )

        assert not sl["sample_has_subgoal"].any()
        sg_comma = pad_masks[:, sl["sg_comma_lo"] : sl["sg_comma_hi"]]
        sg_start = pad_masks[:, sl["sg_start_lo"] : sl["sg_start_hi"]]
        assert torch.all(~sg_comma) and torch.all(~sg_start)

        sg_vid = pad_masks[:, sl["sg_vid_lo"] : sl["sg_vid_hi"]]
        assert torch.all(~sg_vid)

    # ------------------------------------------------------------------ #
    # Subgoal tensors present and not pad — sg block unmasked
    # ------------------------------------------------------------------ #
    def test_subgoal_present_sg_block_unmasked(self):
        """``subgoal0`` / ``subgoal1`` with ``subgoal_is_pad=False`` → per-camera
        masks true; comma + ``Subgoal:`` + SigLIP slots unmasked."""
        bsz = 1
        batch = self._subgoal_batch(bsz, include_tensors=True, subgoal_is_pad=False)
        sg_videos, sg_masks = self._prepare_subgoal(batch)
        assert sg_masks[0].all() and sg_masks[1].all()

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pad_masks, sl = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=sg_videos,
            subgoal_vid_masks=sg_masks,
        )

        assert sl["sample_has_subgoal"].all()
        sg_comma = pad_masks[:, sl["sg_comma_lo"] : sl["sg_comma_hi"]]
        sg_start = pad_masks[:, sl["sg_start_lo"] : sl["sg_start_hi"]]
        assert torch.all(sg_comma) and torch.all(sg_start)

        for ci in range(sl["n_cam"]):
            lo = sl["sg_vid_lo"] + ci * self._N_SG_TOKENS
            hi = lo + self._N_SG_TOKENS
            block = pad_masks[:, lo:hi]
            expected = sg_masks[ci][:, None].expand(bsz, self._N_SG_TOKENS)
            assert torch.equal(block, expected), f"camera {ci} SigLIP pad mask mismatch"

    # ------------------------------------------------------------------ #
    # Tensors present but subgoal_is_pad all True — same as missing keys
    # ------------------------------------------------------------------ #
    def test_subgoal_tensors_all_is_pad_matches_missing_keys(self):
        """Real ``subgoal{k}`` tensors with ``subgoal_is_pad=True`` must match the
        no-key path for ``prepare_subgoal_images`` and for ``embed_prefix`` pad_masks."""
        bsz = 1
        batch_missing = {"state": torch.zeros(bsz, 1, MAX_STATE_DIM)}
        v_miss, m_miss = self._prepare_subgoal(batch_missing)

        batch_pad = self._subgoal_batch(bsz, include_tensors=True, subgoal_is_pad=True)
        v_pad, m_pad = self._prepare_subgoal(batch_pad)

        for i in range(len(v_miss)):
            assert torch.equal(v_miss[i], v_pad[i])
            assert torch.equal(m_miss[i], m_pad[i])

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pm_a, sl = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=v_miss,
            subgoal_vid_masks=m_miss,
        )
        pm_b, _ = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=v_pad,
            subgoal_vid_masks=m_pad,
        )
        assert torch.equal(pm_a, pm_b)

    # ------------------------------------------------------------------ #
    # Mixed batch — per-sample subgoal_is_pad
    # ------------------------------------------------------------------ #
    def test_mixed_batch_subgoal_masking(self):
        """Per-sample ``subgoal_is_pad`` must gate comma / header / vision independently."""
        bsz = 2
        batch = self._subgoal_batch(bsz, include_tensors=True, subgoal_is_pad=torch.tensor([True, False]))
        sg_videos, sg_masks = self._prepare_subgoal(batch)
        assert torch.equal(sg_masks[0], torch.tensor([False, True]))
        assert torch.equal(sg_masks[0], sg_masks[1])

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pad_masks, sl = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=sg_videos,
            subgoal_vid_masks=sg_masks,
        )

        has = sl["sample_has_subgoal"]
        assert torch.equal(has, torch.tensor([False, True]))

        sg_comma = pad_masks[:, sl["sg_comma_lo"] : sl["sg_comma_hi"]]
        assert torch.all(~sg_comma[0]) and torch.all(sg_comma[1])

        sg_start = pad_masks[:, sl["sg_start_lo"] : sl["sg_start_hi"]]
        assert torch.all(~sg_start[0]) and torch.all(sg_start[1])

        sg_vid = pad_masks[:, sl["sg_vid_lo"] : sl["sg_vid_hi"]]
        assert torch.all(~sg_vid[0])
        assert torch.all(sg_vid[1])

    # ------------------------------------------------------------------ #
    # OR across cameras: one camera masked out does not drop the whole sample
    # ------------------------------------------------------------------ #
    def test_subgoal_sample_has_or_across_cameras(self):
        """If either camera mask is true for a row, ``sample_has_subgoal`` is true."""
        bsz = 1
        device = torch.device("cpu")
        batch = {
            "state": torch.zeros(bsz, 1, MAX_STATE_DIM, device=device),
            "subgoal_is_pad": torch.zeros(bsz, dtype=torch.bool, device=device),
            "subgoal0": torch.full((bsz, 3, 224, 224), 0.5, device=device),
            "subgoal1": torch.full((bsz, 3, 224, 224), 0.6, device=device),
        }
        sg_videos, sg_masks = self._prepare_subgoal(batch)
        sg_masks[1][:] = False
        assert sg_masks[0].all() and not sg_masks[1].any()

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pad_masks, sl = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=sg_videos,
            subgoal_vid_masks=sg_masks,
        )
        assert sl["sample_has_subgoal"].all(), "OR across cameras should keep sample active"

        sg_comma = pad_masks[:, sl["sg_comma_lo"] : sl["sg_comma_hi"]]
        assert torch.all(sg_comma)

    # ------------------------------------------------------------------ #
    # Default subgoal_is_pad when omitted (all pad)
    # ------------------------------------------------------------------ #
    def test_omit_subgoal_is_pad_defaults_all_pad(self):
        """Omitted ``subgoal_is_pad`` defaults to all-True (see ``prepare_subgoal_images``)."""
        bsz = 1
        device = torch.device("cpu")
        batch = {
            "state": torch.zeros(bsz, 1, MAX_STATE_DIM, device=device),
            "subgoal0": torch.ones(bsz, 3, 224, 224, device=device) * 0.3,
            "subgoal1": torch.ones(bsz, 3, 224, 224, device=device) * 0.4,
        }
        sg_videos, sg_masks = self._prepare_subgoal(batch)
        assert all(not m.any() for m in sg_masks)

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pad_masks, sl = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=sg_videos,
            subgoal_vid_masks=sg_masks,
        )
        assert not sl["sample_has_subgoal"].any()
        assert torch.all(~pad_masks[:, sl["sg_comma_lo"] : sl["sg_comma_hi"]])

    # ------------------------------------------------------------------ #
    # Parity: identical prepare outputs → identical prefix pad_masks
    # ------------------------------------------------------------------ #
    def test_subgoal_embed_prefix_parity(self):
        """Two ``prepare_subgoal_images`` runs on the same batch yield identical
        ``embed_prefix`` pad_masks."""
        bsz = 1
        batch = self._subgoal_batch(bsz, include_tensors=True, subgoal_is_pad=False)
        v1, m1 = self._prepare_subgoal(batch)
        v2, m2 = self._prepare_subgoal(batch)
        for i in range(len(v1)):
            assert torch.equal(v1[i], v2[i])
            assert torch.equal(m1[i], m2[i])

        rt, rm, mt, mm = self._empty_response_metadata(bsz)
        pm_a, _ = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=v1,
            subgoal_vid_masks=m1,
        )
        pm_b, _ = self._call_embed_prefix_with_subgoals(
            bsize=bsz,
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=v2,
            subgoal_vid_masks=m2,
        )
        assert torch.equal(pm_a, pm_b)

    # ------------------------------------------------------------------ #
    # Subgoals route through ``embed_image`` as 4-D (B, C, H, W) tensors,
    # while the main history video still goes through ``embed_video``.
    # ------------------------------------------------------------------ #
    def test_subgoal_routed_through_embed_image(self):
        bsz = 2
        n_obs_steps = 4
        batch = self._subgoal_batch(
            bsz,
            include_tensors=True,
            subgoal_is_pad=torch.tensor([True, False]),
        )
        sg_images, sg_masks = self._prepare_subgoal(batch)
        rt, rm, mt, mm = self._empty_response_metadata(bsz)

        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=8)
        model.video_encoder.num_frames = n_obs_steps

        video_calls: list[tuple[tuple[int, ...], torch.Tensor | None]] = []
        image_calls: list[tuple[int, ...]] = []

        def _capturing_embed_video(video, obs_history_is_pad=None):
            t = video.shape[1]
            max_t = model.video_encoder.num_frames
            if not 1 <= t <= max_t:
                raise ValueError(f"Expected 1 <= T <= {max_t}; got {t}.")
            video_calls.append((tuple(video.shape), obs_history_is_pad))
            return torch.zeros(video.shape[0], 6, 8, dtype=torch.bfloat16)

        def _capturing_embed_image(image):
            image_calls.append(tuple(image.shape))
            return torch.zeros(image.shape[0], 4, 8, dtype=torch.bfloat16)

        model.embed_video = _capturing_embed_video
        model.paligemma_with_expert.embed_image = _capturing_embed_image

        _legacy_embed_prefix(
            model,
            videos=[torch.zeros(bsz, n_obs_steps, 3, 8, 8)],
            vid_masks=[torch.ones(bsz, dtype=torch.bool)],
            lang_tokens=torch.zeros(bsz, 5, dtype=torch.long),
            lang_masks=torch.ones(bsz, 5, dtype=torch.bool),
            state=torch.zeros(bsz, n_obs_steps, 8),
            response_tokens=rt,
            response_masks=rm,
            metadata_tokens=mt,
            metadata_masks=mm,
            subgoal_videos=sg_images,
            subgoal_vid_masks=sg_masks,
        )

        # Main history video → one embed_video call (T=n_obs_steps); subgoal
        # images → embed_image, one call per subgoal camera, each a 4-D
        # (B, C, H, W) tensor. Sample 1 has a real subgoal, so the cross-rank
        # OR of the mask is True (here single-process, so just the local
        # ``.any()``) and the image tower runs for every camera; the padded
        # sample 0 is still embedded then masked downstream.
        assert len(video_calls) == 1
        assert video_calls[0][0] == (bsz, n_obs_steps, 3, 8, 8)
        assert len(image_calls) == len(sg_images)
        for shape in image_calls:
            assert shape == (bsz, 3, 224, 224), shape

    # ------------------------------------------------------------------ #
    # Image-tower skip via _embed_item's local fallback: called directly with
    # run_image_tower=None (no process group), the image branch decides from
    # the rank-local ``.any()`` and fires no collective. The hoisted cross-rank
    # decision is covered by the _global_run_image_tower / lockstep tests below.
    # ------------------------------------------------------------------ #
    def test_all_padded_subgoal_skips_image_tower(self):
        """All-padded subgoal mask → ``embed_image`` is not called and the image
        block is zeros of width ``num_image_tokens``."""
        bsz, h = 3, 8
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=h)
        n_img_tokens = model.paligemma_with_expert.config.paligemma_config.text_config.num_image_tokens

        calls: list[tuple[int, ...]] = []

        def _exploding_embed_image(image):
            calls.append(tuple(image.shape))
            raise AssertionError("embed_image must be skipped when the subgoal is all-padded")

        model.paligemma_with_expert.embed_image = _exploding_embed_image

        item = ContextItem(
            data=torch.full((bsz, 3, 224, 224), 0.5),
            item_type="image",
            pad_mask=torch.zeros(bsz, dtype=torch.bool),
            attention="continue",
        )
        emb, expanded_mask = model._embed_item(item)

        assert calls == [], "image tower should not run for an all-padded subgoal"
        assert emb.shape == (bsz, n_img_tokens, h)
        assert emb.dtype == torch.bfloat16
        assert not emb.any(), "skipped image embedding must be all zeros"
        assert expanded_mask.shape == (bsz, n_img_tokens)
        assert not expanded_mask.any(), "all-padded mask stays all-False after expansion"

    def test_some_real_subgoal_runs_image_tower(self):
        """A mask with any real sample runs ``embed_image`` (no skip)."""
        bsz, h = 3, 8
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=h)

        calls: list[tuple[int, ...]] = []

        def _capturing_embed_image(image):
            calls.append(tuple(image.shape))
            return torch.full((image.shape[0], 4, h), 7.0, dtype=torch.bfloat16)

        model.paligemma_with_expert.embed_image = _capturing_embed_image

        item = ContextItem(
            data=torch.full((bsz, 3, 224, 224), 0.5),
            item_type="image",
            pad_mask=torch.tensor([False, True, False]),
            attention="continue",
        )
        emb, expanded_mask = model._embed_item(item)

        assert calls == [(bsz, 3, 224, 224)], "image tower should run when any sample is real"
        assert emb.shape == (bsz, 4, h)
        assert emb.eq(7.0).all(), "running path must return the real embed_image output, not zeros"
        # Per-sample mask expands across the 4 image tokens.
        assert expanded_mask.shape == (bsz, 4)
        assert expanded_mask[1].all() and not expanded_mask[0].any() and not expanded_mask[2].any()

    def test_run_image_tower_override_wins_over_local_mask(self):
        """The decision passed from ``embed_prefix`` overrides the local mask:
        ``run_image_tower=False`` skips even a real sample; ``True`` runs even an
        all-padded one. This is the contract that lets the cross-rank decision
        actually control the per-item branch."""
        bsz, h = 3, 8
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=h)
        n_img_tokens = model.paligemma_with_expert.config.paligemma_config.text_config.num_image_tokens
        calls: list[tuple[int, ...]] = []

        def _capturing_embed_image(image):
            calls.append(tuple(image.shape))
            return torch.full((image.shape[0], 4, h), 1.0, dtype=torch.bfloat16)

        model.paligemma_with_expert.embed_image = _capturing_embed_image

        # Real sample but forced skip → zeros, no call.
        real = ContextItem(
            data=torch.full((bsz, 3, 224, 224), 0.5),
            item_type="image",
            pad_mask=torch.tensor([False, True, False]),
            attention="continue",
        )
        emb_skip, _ = model._embed_item(real, run_image_tower=False)
        assert calls == [], "run_image_tower=False must skip even a real sample"
        assert emb_skip.shape == (bsz, n_img_tokens, h) and not emb_skip.any()

        # All-padded but forced run → real output, one call.
        padded = ContextItem(
            data=torch.full((bsz, 3, 224, 224), 0.5),
            item_type="image",
            pad_mask=torch.zeros(bsz, dtype=torch.bool),
            attention="continue",
        )
        emb_run, _ = model._embed_item(padded, run_image_tower=True)
        assert calls == [(bsz, 3, 224, 224)], "run_image_tower=True must run even an all-padded sample"
        assert emb_run.shape == (bsz, 4, h)

    def test_global_run_image_tower_local_decision(self):
        """Single-process (no process group): ``_global_run_image_tower`` returns
        the local OR over image items — True if any camera mask is real, False
        for all-padded, and False when the prefix has no image item at all."""
        bsz, h = 2, 8
        model = TestPI07PaligemmaLowLevelResponseEmbedding._make_mock_model(hidden_size=h)

        def _img(mask):
            return ContextItem(
                data=torch.zeros(bsz, 3, 224, 224), item_type="image", pad_mask=mask, attention="continue"
            )

        text = ContextItem(
            data=torch.zeros(bsz, 3, dtype=torch.long),
            item_type="text",
            pad_mask=torch.ones(bsz, 3, dtype=torch.bool),
            attention="continue",
        )
        # Two cameras, one with a real sample → run.
        assert model._global_run_image_tower(
            [text, _img(torch.zeros(bsz, dtype=torch.bool)), _img(torch.tensor([False, True]))]
        )
        # All cameras all-padded → skip.
        assert not model._global_run_image_tower([text, _img(torch.zeros(bsz, dtype=torch.bool))])
        # No image item at all → skip (and no crash deriving the device).
        assert not model._global_run_image_tower([text])

    # ------------------------------------------------------------------ #
    # DDP lockstep: the hoisted ``_global_run_image_tower`` collective must make
    # the run/skip decision follow the *global* OR (not each rank's local
    # ``.any()``), and must detect cross-rank presence divergence loudly —
    # otherwise ranks fire a different number of collectives and hang at NCCL.
    # Verified over a 2-rank gloo group on CPU (no GPU needed).
    # ------------------------------------------------------------------ #
    @pytest.mark.slow
    def test_ddp_lockstep_image_tower_skip(self):
        """Divergent per-rank masks → both ranks RUN (global OR); all-padded →
        both SKIP; one rank missing the image item → loud RuntimeError, not a
        silent collective-count desync."""
        import torch.multiprocessing as mp

        if not torch.distributed.is_available():
            pytest.skip("torch.distributed is not available")

        for scenario in ("divergent", "all_padded", "presence_divergence"):
            mp.spawn(_ddp_image_skip_worker, args=(2, _find_free_port(), scenario), nprocs=2, join=True)


class TestPI07PaligemmaLowLevelExecutionHorizon:
    """Regression coverage for ``n_action_steps`` as a short execution horizon.

    ``chunk_size`` is the trained prediction horizon (always decoded);
    ``n_action_steps`` (<= chunk_size) is how many actions are executed before
    re-querying. ``n_action_steps < chunk_size`` used to broadcast-crash the
    denoise/MSE paths and trip the queue assert; these CPU tests guard that
    path (the GPU pipeline test only exercises ``n_action_steps == chunk_size``).
    """

    @staticmethod
    def _config(chunk_size=10, n_action_steps=3, max_delay=0):
        return PI07PaligemmaLowLevelConfig(
            n_obs_steps=1,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            max_delay=max_delay,
            max_state_dim=MAX_STATE_DIM,
            max_action_dim=MAX_ACTION_DIM,
        )

    def test_guard_rejects_short_horizon_with_delay(self):
        # A shortened execution horizon is not compatible with the real-time
        # delay prefix path; the config must reject it loudly.
        with pytest.raises(ValueError, match="max_delay"):
            self._config(chunk_size=10, n_action_steps=3, max_delay=2)

    def test_guard_allows_short_horizon_without_delay(self):
        cfg = self._config(chunk_size=10, n_action_steps=3, max_delay=0)
        assert (cfg.chunk_size, cfg.n_action_steps, cfg.max_delay) == (10, 3, 0)

    def test_guard_allows_full_horizon_with_delay(self):
        # n_action_steps == chunk_size keeps the real-time-delay path available.
        cfg = self._config(chunk_size=10, n_action_steps=10, max_delay=2)
        assert cfg.max_delay == 2

    def test_select_action_executes_first_n_then_requeries(self):
        """Model decodes the full ``chunk_size`` chunk, but ``select_action``
        executes only the first ``n_action_steps`` before re-querying.
        """
        chunk_size, n_steps, bsz = 10, 3, 2
        cfg = self._config(chunk_size=chunk_size, n_action_steps=n_steps, max_delay=0)

        policy = object.__new__(PI07PaligemmaLowLevelPolicy)
        policy.config = cfg
        policy.eval = lambda: None  # bypass nn.Module.eval (no __init__ was run)
        PI07PaligemmaLowLevelPolicy.reset(policy)

        calls = {"n": 0}

        def fake_sample_actions(batch, noise=None, action_prefix=None, delay=None):
            # Return a full chunk_size chunk; element value encodes
            # (call_index * 1000 + timestep) so we can assert which timesteps run.
            calls["n"] += 1
            ts = torch.arange(chunk_size, dtype=torch.float32).reshape(1, chunk_size, 1)
            return (calls["n"] * 1000 + ts).expand(bsz, chunk_size, MAX_ACTION_DIM).clone()

        policy.sample_actions = fake_sample_actions
        batch = {"state": torch.zeros(bsz, MAX_STATE_DIM)}

        # First n_steps actions all come from a single decode (call 1), in order.
        acts = [PI07PaligemmaLowLevelPolicy.select_action(policy, batch) for _ in range(n_steps)]
        assert calls["n"] == 1
        assert [tuple(a.shape) for a in acts] == [(bsz, MAX_ACTION_DIM)] * n_steps
        assert [a[0, 0].item() for a in acts] == [1000.0, 1001.0, 1002.0]

        # Queue drained after n_action_steps -> next call re-queries (call 2).
        a_next = PI07PaligemmaLowLevelPolicy.select_action(policy, batch)
        assert calls["n"] == 2
        assert a_next[0, 0].item() == 2000.0


# CPU-only tests for the metadata-string assembly loop in
# ``PI07PaligemmaLowLevelPolicy.prepare_metadata``. Pin the contract that
# ``fps`` is tokenized with the ``"FPS: N, "`` header (uppercase to match
# the sibling ``Speed:``/``Quality:``/``Mistake:``/``Robot:``/``Control:``
# labels) positioned between ``Robot:`` and ``Control:`` and omitted entirely
# when ``fps_is_pad`` is True (or when the keys are missing —
# ``_hydrate_metadata_batch`` defaults them to all-padded).


def _pg_ll_make_fake_policy(metadata_max_length: int = 4):
    """Construct a stubbed PI07PaligemmaLowLevelPolicy for prepare_metadata tests.

    ``prepare_metadata`` calls ``self._hydrate_metadata_batch(batch)`` first,
    so a bare ``SimpleNamespace`` doesn't work — we use
    ``object.__new__(PI07PaligemmaLowLevelPolicy)`` to get a real instance
    with the class methods bound, then stub only the tokenizer + config to
    skip the PaliGemma backbone download.
    """
    import types

    captured: list[str] = []

    def stub_call(metadata, **kwargs):
        captured.extend(metadata)
        batch_size = len(metadata)
        max_length = kwargs.get("max_length", 4)
        return {
            "input_ids": torch.zeros((batch_size, max_length), dtype=torch.long),
            "attention_mask": torch.zeros((batch_size, max_length), dtype=torch.long),
        }

    tokenizer = types.SimpleNamespace()
    tokenizer.__call__ = stub_call

    policy = object.__new__(PI07PaligemmaLowLevelPolicy)
    policy.language_tokenizer = tokenizer
    policy.config = types.SimpleNamespace(metadata_max_length=metadata_max_length)
    return policy, captured


class TestPaligemmaLowLevelFpsSegment:
    def test_fps_present_emits_uppercase_segment(self):
        policy, captured = _pg_ll_make_fake_policy()
        batch_size = 2
        batch = {
            "state": torch.zeros(batch_size, 1),
            "fps": torch.tensor([30, 50], dtype=torch.long),
            "fps_is_pad": torch.tensor([False, False]),
        }
        # _hydrate_metadata_batch will fill in speed/quality/mistake/* defaults
        # (all padded) and empty robot_type/control_mode lists.

        PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

        assert len(captured) == batch_size
        for line, fps in zip(captured, [30, 50], strict=True):
            assert line.startswith("Metadata: ")
            assert f"FPS: {fps}, " in line
            assert "fps:" not in line and "Fps:" not in line

    def test_fps_padded_omits_segment(self):
        policy, captured = _pg_ll_make_fake_policy()
        batch_size = 1
        batch = {
            "state": torch.zeros(batch_size, 1),
            "fps": torch.tensor([30], dtype=torch.long),
            "fps_is_pad": torch.tensor([True]),
            "robot_type": ["franka"],
        }

        PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

        assert "FPS:" not in captured[0]
        assert ", ," not in captured[0]
        assert "Robot: franka, " in captured[0]

    def test_fps_key_absent_omits_segment(self):
        """``_hydrate_metadata_batch`` defaults ``fps_is_pad`` to True when
        absent, so missing keys collapse to the same "no segment" behaviour
        as explicit-pad."""
        policy, captured = _pg_ll_make_fake_policy()
        batch_size = 3
        batch = {"state": torch.zeros(batch_size, 1)}

        PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

        for line in captured:
            assert "FPS:" not in line

    def test_fps_slots_between_robot_and_control(self):
        policy, captured = _pg_ll_make_fake_policy()
        batch_size = 1
        batch = {
            "state": torch.zeros(batch_size, 1),
            "fps": torch.tensor([20], dtype=torch.long),
            "fps_is_pad": torch.tensor([False]),
            "robot_type": ["franka"],
            "control_mode": ["joint"],
        }

        PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

        line = captured[0]
        robot_idx = line.index("Robot: ")
        fps_idx = line.index("FPS: 20")
        control_idx = line.index("Control: ")
        assert robot_idx < fps_idx < control_idx, f"Expected Robot < FPS < Control ordering in {line!r}"

    def test_fps_default_torch_long(self):
        """fps is hydrated with ``torch.long`` defaults; assert ``_row_long``
        preserves the dtype through the broadcast path."""
        policy, captured = _pg_ll_make_fake_policy()
        # Scalar-broadcast fps (one element, fan out to (B,)).
        batch_size = 3
        batch = {
            "state": torch.zeros(batch_size, 1),
            "fps": torch.tensor([25], dtype=torch.long),
            "fps_is_pad": torch.tensor([False]),
        }

        PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

        # All three samples get FPS: 25 from the scalar-broadcast.
        assert len(captured) == 3
        for line in captured:
            assert "FPS: 25, " in line

    def test_fps_mixed_pad_across_batch(self):
        """Per-sample ``fps_is_pad`` must be honored row-by-row — production
        path for heterogeneous LeRobot + VQA mixtures, where the VQA pad
        row (``fps=0, fps_is_pad=True``) sits next to a LeRobot row
        (``fps=30, fps_is_pad=False``) in the same batch.
        """
        policy, captured = _pg_ll_make_fake_policy()
        batch = {
            "state": torch.zeros(2, 1),
            "fps": torch.tensor([30, 0], dtype=torch.long),
            "fps_is_pad": torch.tensor([False, True]),
        }

        PI07PaligemmaLowLevelPolicy.prepare_metadata(policy, batch)

        assert "FPS: 30, " in captured[0]
        assert "FPS:" not in captured[1]


class TestSkipNormalizationWeights:
    """Smoke test that ``PI07PaligemmaLowLevelConfig`` still carries the
    inherited ``skip_normalization_weights`` field.

    Cross-policy tests for the field default and keyword-settability live
    in :mod:`tests.policies.test_pretrained_skip_normalization`. The shared
    predicate / inf-guard tests live in
    :mod:`tests.policies.test_normalize_helpers`.
    """

    def test_default_is_false(self):
        cfg = PI07PaligemmaLowLevelConfig()
        assert cfg.skip_normalization_weights is False

    def test_can_be_set_true(self):
        cfg = PI07PaligemmaLowLevelConfig(skip_normalization_weights=True)
        assert cfg.skip_normalization_weights is True
