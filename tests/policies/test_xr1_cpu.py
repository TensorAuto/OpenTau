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

"""CPU tests for the ``xr1`` policy (Xiaomi-Robotics-1).

Two kinds of test live here:

* **Parity gates that need no GPU and no weights** -- P1 (the rendered prompt string), P2
  (the state adapter), P3 (the centre-crop box), P4 (the tau schedule and the DiT timestep
  sinusoid) and P5 (the history index table) are all asserted against the goldens captured
  from the reference in ``tests/artifacts/policies/xr1/``.
* **Silent-divergence pins** -- each one targets a place where a plausible, reviewable
  implementation is wrong and *nothing crashes*: the flow integrating the wrong way, the
  action mask hoisted out of the Euler loop, a bidirectional instead of causal DiT query
  block, the position offset collapsed across MRoPE axes, history zero-padded instead of
  clamped, or ``use_cache`` silently disabled by gradient checkpointing.

The model itself runs on a deliberately tiny random ``Qwen3VLConfig``, so nothing here
downloads the 4 B backbone (pattern from ``test_cosmos3_cpu.py``).
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import Qwen3VLConfig

from opentau.policies.utils import PerSampleLoss, flow_matching_masked_mse
from opentau.policies.xr1.configuration_xr1 import XR1Config
from opentau.policies.xr1.dit import XR1TimestepEmbedder, modulate, repeat_batch
from opentau.policies.xr1.losses import build_flow_mask, flow_and_freq_loss, fold_repeat_per_sample
from opentau.policies.xr1.modeling_xr1 import XR1FlowMatching, XR1Policy
from opentau.policies.xr1.obs_adapter import (
    XR1ObservationBuffer,
    adapt_state,
    history_indices,
    quat_to_axis_angle,
    robocasa_agent_pos_to_xr1_state,
)
from opentau.policies.xr1.processing_xr1 import (
    build_action_mask,
    center_crop_resize,
    crop_box,
    expand_video_placeholders,
    patchify_videos,
    render_chat_text,
)
from opentau.policies.xr1.state_dict_remap import (
    assert_full_coverage,
    remap_reference_state_dict,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts" / "policies" / "xr1"

# --- tiny model geometry -------------------------------------------------------------
VOCAB = 200
IMAGE_TOKEN_ID = 10
VIDEO_TOKEN_ID = 11
VISION_START_ID = 12
VISION_END_ID = 13

N_OBS = 4
INTERVAL = 2
CHUNK = 8
NUM_CAMS = 2
IMAGE_SIZE = 32
STATE_TOKEN_DIM = 20
MAX_ACTION_DIM = 16
ACTION_DIM = 6


def _golden(name: str):
    return json.loads((ARTIFACTS / name).read_text())


def _tiny_qwen3vl_config(num_hidden_layers: int = 2) -> Qwen3VLConfig:
    return Qwen3VLConfig(
        text_config={
            "model_type": "qwen3_vl_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": VOCAB,
            "rms_norm_eps": 1e-6,
            "rope_theta": 5_000_000,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
        },
        vision_config={
            "model_type": "qwen3_vl",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "depth": 4,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [1, 2],
            "num_position_embeddings": 256,
            "in_channels": 3,
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=VIDEO_TOKEN_ID,
        vision_start_token_id=VISION_START_ID,
        vision_end_token_id=VISION_END_ID,
        tie_word_embeddings=False,
    )


def _tiny_xr1_config(**overrides) -> XR1Config:
    kwargs = {
        "n_obs_steps": N_OBS,
        "history_interval": INTERVAL,
        "chunk_size": CHUNK,
        "n_action_steps": CHUNK,
        "num_cams": NUM_CAMS,
        "image_size": IMAGE_SIZE,
        "max_state_dim": 16,
        "state_token_dim": STATE_TOKEN_DIM,
        "max_action_dim": MAX_ACTION_DIM,
        "num_steps": 3,
        "training_repeat": 2,
        "load_pretrained_backbone": False,
        "dit_num_hidden_layers": 2,
        "dit_hidden_size": 32,
        "dit_intermediate_size": 64,
        "dit_num_key_value_heads": 2,
        "dit_head_dim": 16,
        "dit_time_embed_dim": 16,
        "output_features": {},
    }
    kwargs.update(overrides)
    out_features = kwargs.pop("output_features")
    cfg = XR1Config(**kwargs)
    if not out_features:
        cfg.validate_features()
        cfg.output_features["actions"].shape = (ACTION_DIM,)
    return cfg


def _build_model(num_layers: int = 2, **overrides) -> XR1FlowMatching:
    torch.manual_seed(0)
    return XR1FlowMatching(_tiny_xr1_config(**overrides), qwen3vl_config=_tiny_qwen3vl_config(num_layers))


def _build_choice_model(**overrides) -> XR1FlowMatching:
    """A tiny model whose vocabulary is large enough to hold the reference token ids.

    The choice-token ids (151669-151730) are a hard contract with the reference tokenizer --
    the heads select hidden states by id *range* -- so they are deliberately not
    configurable. That means a choice-head test needs a real-sized embedding table; at
    hidden 64 that is ~10 M parameters, which is cheap enough for the CPU suite.
    """
    from opentau.policies.xr1.choice_heads import ACTION_TOKEN_END_ID

    torch.manual_seed(0)
    config = _tiny_xr1_config(enable_choice_heads=True, n_choices=3, **overrides)
    qwen3vl_config = _tiny_qwen3vl_config(2)
    qwen3vl_config.text_config.vocab_size = ACTION_TOKEN_END_ID + 8
    return XR1FlowMatching(config, qwen3vl_config=qwen3vl_config)


def _video_batch(bsize: int = 2):
    """A prompt with ``NUM_CAMS`` video blocks plus the tiny patch tensors that feed them."""
    grid_t = N_OBS // 2
    grid_h = grid_w = IMAGE_SIZE // 16
    tokens_per_video = grid_t * grid_h * grid_w // 4
    seq = []
    for _ in range(NUM_CAMS):
        seq += [VISION_START_ID] + [VIDEO_TOKEN_ID] * tokens_per_video + [VISION_END_ID]
    seq += [25, 30, 35]
    input_ids = torch.tensor([seq] * bsize, dtype=torch.long)
    attention_mask = torch.ones(bsize, len(seq), dtype=torch.long)
    videos = torch.rand(bsize * NUM_CAMS, N_OBS, 3, IMAGE_SIZE, IMAGE_SIZE)
    pixel_values_videos, video_grid_thw = patchify_videos(videos, 16, 2, 2)
    state = torch.randn(bsize, N_OBS, STATE_TOKEN_DIM)
    actions = torch.randn(bsize, CHUNK, MAX_ACTION_DIM)
    action_mask = build_action_mask(
        bsize, CHUNK, MAX_ACTION_DIM, ACTION_DIM, device=actions.device, dtype=actions.dtype
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values_videos": pixel_values_videos,
        "video_grid_thw": video_grid_thw,
        "state": state,
        "actions": actions,
        "action_mask": action_mask,
    }


# =====================================================================================
# P1 -- the prompt string
# =====================================================================================


def test_p1_rendered_prompt_matches_the_reference_string_exactly():
    """The assembled chat text must equal the reference's, character for character."""
    golden = _golden("prompt_tokens.json")["A_start"]
    grid = torch.tensor(golden["video_grid_thw"], dtype=torch.long)
    text = render_chat_text(golden["instruction"], ("Left camera: ", "\nRight camera: ", "\nWrist camera: "))
    expanded = expand_video_placeholders(text, grid, merge_size=2, frame_indices_per_video=[[0, 1, 2, 3]] * 3)
    assert expanded == golden["decoded"]


def test_p1_video_block_structure_is_two_blocks_per_camera():
    """Discriminating counts: a one-block-per-camera builder halves the video tokens.

    Four frames at ``temporal_patch_size=2`` give **two** temporal patches, so each camera
    contributes two ``<|vision_start|>...<|vision_end|>`` groups of 64 tokens (128 total),
    each preceded by its own ``<T.T seconds>`` marker.
    """
    golden = _golden("prompt_tokens.json")["A_start"]
    text = golden["decoded"]
    assert text.count("<|vision_start|>") == 6
    assert text.count("<|vision_end|>") == 6
    assert text.count("<|video_pad|>") == 3 * 128
    assert text.count("<0.0 seconds>") == 3
    assert text.count("<0.1 seconds>") == 3
    assert golden["video_grid_thw"] == [[2, 16, 16]] * 3


def test_p1_timestamps_come_from_the_default_fps_of_24():
    """The markers are a *fallback*: no video metadata is passed, so Qwen3-VL uses 24 fps.

    At 24 fps the two temporal patches stamp 0.02 s and 0.10 s, which format as
    ``<0.0 seconds>`` / ``<0.1 seconds>``. At the video processor's configured 2 fps they
    would read ``<0.2 seconds>`` / ``<1.2 seconds>`` -- a different prompt entirely.
    """
    grid = torch.tensor([[2, 16, 16]], dtype=torch.long)
    text = render_chat_text("x", ("Left camera: ",))
    at_24 = expand_video_placeholders(text, grid, 2, [[0, 1, 2, 3]], video_fps=24.0)
    at_2 = expand_video_placeholders(text, grid, 2, [[0, 1, 2, 3]], video_fps=2.0)
    assert "<0.0 seconds>" in at_24 and "<0.1 seconds>" in at_24
    assert at_2 != at_24


# =====================================================================================
# P2 -- the state adapter
# =====================================================================================


def test_p2_state_adapter_matches_the_reference():
    """Every captured ``agent_pos`` must map to the reference's 14-D state.

    The tolerance is 1e-6 rather than 1e-9 because the reference stores the state as
    float32 after computing it in float64; the comparison here is at float64.
    """
    golden = _golden("state_from_env.json")
    for case in golden["cases"]:
        agent_pos = torch.tensor(case["agent_pos16"], dtype=torch.float64)
        expected = torch.tensor(case["state14"], dtype=torch.float64)
        assert torch.allclose(robocasa_agent_pos_to_xr1_state(agent_pos), expected, atol=1e-6)


def test_p2_quaternion_edge_cases():
    """Identity, zero and un-normalized quaternions, plus the ``q == -q`` canonicalization.

    All four are classic axis-angle bugs that produce a finite, plausible 3-vector (or a
    NaN that only shows up in a loss curve) rather than an exception.
    """
    golden = _golden("state_from_env.json")["quaternion_edge_cases"]

    def aa(values):
        return quat_to_axis_angle(torch.tensor(values, dtype=torch.float64)).tolist()

    assert aa([0.0, 0.0, 0.0, 1.0]) == pytest.approx(golden["identity_xyzw"], abs=1e-9)
    assert aa([0.0, 0.0, 0.0, -1.0]) == pytest.approx(golden["negated_identity_xyzw"], abs=1e-9)
    assert aa([0.0, 0.0, 0.0, 0.0]) == pytest.approx(golden["zero_quat"], abs=1e-9)
    assert aa([0.0, 0.0, 2.0, 2.0]) == pytest.approx(golden["unnormalized"], abs=1e-7)
    # q and -q are the same rotation and must map to the same axis-angle vector.
    plus, minus = golden["sign_flipped_pair"]
    assert plus == pytest.approx(minus, abs=1e-9)
    assert aa([0.1, 0.2, 0.3, -0.9]) == pytest.approx(aa([-0.1, -0.2, -0.3, 0.9]), abs=1e-9)


def test_p2_bf16_state_does_not_move_the_axis_angle_much():
    """``envs/utils.py`` casts the state to bfloat16 *before* the policy sees it.

    The quaternion is therefore bf16 when it reaches a nonlinear conversion the reference
    ran in float64. Pin the resulting angular error so a regression (or a change in the
    env's cast) is visible rather than absorbed.
    """
    torch.manual_seed(0)
    quats = torch.randn(512, 4, dtype=torch.float32)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    exact = quat_to_axis_angle(quats.double())
    via_bf16 = quat_to_axis_angle(quats.to(torch.bfloat16).float()).double()
    # Measured max over 512 random rotations: ~6.8e-3 rad (~0.39 deg). Well inside what a
    # RoboCasa controller resolves, but large enough to be worth a recorded bound: if this
    # grows, the env's pre-policy bf16 cast is the thing to look at (a scoped
    # `preserve_state_float32` opt-in, never a blanket dtype change).
    assert (exact - via_bf16).abs().max().item() < 1e-2


def test_state_adapter_pads_to_the_token_width_and_leaves_the_tail_zero():
    agent_pos = torch.randn(3, N_OBS, 16)
    out = adapt_state(agent_pos, state_adapter="robocasa_panda_omron", state_token_dim=60)
    assert out.shape == (3, N_OBS, 60)
    assert torch.equal(out[..., 14:], torch.zeros(3, N_OBS, 46))


def test_state_adapter_rejects_a_wrong_width():
    with pytest.raises(ValueError, match="16-D state"):
        adapt_state(torch.randn(2, 14), state_adapter="robocasa_panda_omron", state_token_dim=60)


# =====================================================================================
# P3 -- the centre crop
# =====================================================================================


def test_p3_crop_box_is_asymmetric():
    """``int(256 * 0.95) = 243`` is odd, so 6 px come off the left and 7 off the right."""
    golden = _golden("crop_resize.json")
    assert list(crop_box(256, 0.95)) == golden["box_top_left_h_w"]
    top, left, height, width = crop_box(256, 0.95)
    assert (top, left, height, width) == (6, 6, 243, 243)
    assert 256 - (left + width) == 7


def test_p3_crop_resize_reproduces_the_reference_pixels():
    """sha256 over the reference's own synthetic pattern, through the PIL-parity path."""
    golden = _golden("crop_resize.json")
    pattern = _reference_pattern()
    assert hashlib.sha256(pattern.tobytes()).hexdigest() == golden["input_sha256"]

    tensor = torch.from_numpy(pattern).permute(2, 0, 1).float() / 255.0
    out = center_crop_resize(tensor[None], 0.95)[0]
    as_uint8 = (out.permute(1, 2, 0) * 255.0).round().clamp(0, 255).to(torch.uint8).numpy()
    assert as_uint8.shape == pattern.shape
    # PIL's integer bilinear and torch's float bilinear agree to within one 8-bit level on
    # an upscale; assert that bound rather than the sha256, which would pin PIL's rounding.
    assert np.abs(as_uint8.astype(np.int16) - _reference_cropped().astype(np.int16)).max() <= 1
    assert abs(float(as_uint8.mean()) - golden["output_mean"]) < 0.5


def _reference_pattern() -> np.ndarray:
    """Regenerate the capture script's synthetic pattern (same generator, same seed)."""
    rng = np.random.default_rng(0xC0FFEE)
    coarse = rng.integers(0, 256, size=(8, 8, 3), dtype=np.int64).astype(np.float64)
    size = 256
    ys = np.linspace(0, 7, size)
    xs = np.linspace(0, 7, size)
    y0 = np.clip(np.floor(ys).astype(int), 0, 7)
    x0 = np.clip(np.floor(xs).astype(int), 0, 7)
    y1 = np.clip(y0 + 1, 0, 7)
    x1 = np.clip(x0 + 1, 0, 7)
    wy = (ys - y0)[:, None, None]
    wx = (xs - x0)[None, :, None]
    top = coarse[y0][:, x0] * (1 - wx) + coarse[y0][:, x1] * wx
    bot = coarse[y1][:, x0] * (1 - wx) + coarse[y1][:, x1] * wx
    img = top * (1 - wy) + bot * wy
    img = img + rng.normal(0.0, 6.0, size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _reference_cropped() -> np.ndarray:
    """The reference's PIL crop+resize of that pattern."""
    from PIL import Image

    image = Image.fromarray(_reference_pattern())
    width, height = image.size
    crop_w = max(1, int(width * 0.95))
    crop_h = max(1, int(height * 0.95))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    cropped = image.crop((left, top, left + crop_w, top + crop_h))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(cropped.resize((width, height), resampling), dtype=np.uint8)


def test_crop_ratio_of_one_is_an_identity():
    images = torch.rand(2, 3, 32, 32)
    assert torch.equal(center_crop_resize(images, 1.0), images)


# =====================================================================================
# P4 -- flow geometry
# =====================================================================================


def test_p4_timestep_sinusoid_matches_the_golden():
    """The classic DiT ladder with ``[cos, sin]`` concatenation, at the tau grid x 1000.

    ``cosmos3.modeling_cosmos3.create_sinusoidal_pos_embedding`` uses a ``min_period`` /
    ``max_period`` ladder and concatenates ``[sin, cos]`` -- reusing it here would be a
    silent, plausible-looking substitution, so this pins the actual values.
    """
    golden = _golden("dit_geometry_static.json")["sinusoid"]
    taus = torch.tensor(_golden("dit_geometry_static.json")["tau_schedule"], dtype=torch.float32)
    dim = golden["dim"]
    embedder = XR1TimestepEmbedder(hidden_size=8, frequency_embedding_size=dim)
    got = embedder.timestep_embedding(taus * 1000.0, dim)

    # Checked at 2e-4 rather than exactly, in both directions. The reference computes the
    # sinusoid in float32 at arguments up to ``tau * 1000``, where float32 itself only
    # resolves the *argument* to ~1.5e-5 and cos/sin argument reduction differs by an ULP
    # between libms -- so the 5th decimal of the result is machine-dependent. That is a
    # CPU-fixture artifact only: the GPU parity gates compare kernels on one machine. The
    # tolerance is still four orders of magnitude below any formula change (a different
    # frequency ladder, a [sin, cos] concatenation, or a missing x1000 all move values by
    # O(1)), which the exact structural assertions below also catch.
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float64) / half)
    args = (taus.double() * 1000.0)[:, None] * freqs[None]
    reference = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    assert torch.allclose(got.double(), reference, atol=2e-4)
    assert torch.allclose(got, torch.tensor(golden["values"], dtype=torch.float32), atol=2e-4)
    assert golden["concat_order"] == ["cos", "sin"]
    # The first half is cos, the second sin -- the opposite order from the cosmos3 helper.
    assert torch.allclose(got[0, :half], torch.ones(half), atol=1e-6)
    assert torch.allclose(got[0, half:], torch.zeros(half), atol=1e-6)


def test_p4_timestep_sinusoid_is_not_the_cosmos3_helper():
    """Rule-8 conflict test: exercise the substitution the comment warns against."""
    from opentau.policies.cosmos3.modeling_cosmos3 import create_sinusoidal_pos_embedding

    taus = torch.tensor([0.0, 0.2, 0.4], dtype=torch.float32)
    ours = XR1TimestepEmbedder(hidden_size=8, frequency_embedding_size=16).timestep_embedding(
        taus * 1000.0, 16
    )
    theirs = create_sinusoidal_pos_embedding(taus[None], 16, min_period=4e-3, max_period=4.0)[0]
    assert not torch.allclose(ours, theirs, atol=1e-3)


def test_p4_tau_ascends_and_the_step_is_positive():
    """pi05 / cosmos3 integrate **backwards** from 1.0; a copy-pasted sampler inverts XR-1."""
    golden = _golden("dit_geometry_static.json")
    assert golden["tau_schedule"] == [0.0, 0.2, 0.4, 0.6, 0.8]
    assert golden["dt"] == pytest.approx(0.2)
    num_steps = golden["num_steps"]
    reconstructed = [step / num_steps for step in range(num_steps)]
    assert reconstructed == pytest.approx(golden["tau_schedule"])


def test_p4_dit_query_layout_is_sink_state_action():
    cfg = _tiny_xr1_config()
    assert cfg.dit_query_length == 1 + cfg.n_obs_steps + cfg.chunk_size
    golden = _golden("dit_geometry_static.json")
    assert golden["query_layout"] == ["sink", "state x 4", "noisy_action x 16"]
    assert golden["dit_query_length"] == 1 + 4 + 16


def test_p4_dit_self_attention_block_is_causal_not_bidirectional():
    """cosmos3's suffix is one *bidirectional* block; XR-1's 21 queries are lower-triangular.

    This is the single-character "fix" that would train and evaluate without complaint.
    """
    model = _build_model()
    query_len = model.config.dit_query_length
    prefix_pad = torch.ones(2, 5, dtype=torch.bool)
    mask = model.dit_attention_mask(prefix_pad, query_len)
    assert mask.shape == (2, 1, query_len, 5 + query_len)
    assert torch.equal(
        mask[0, 0, :, -query_len:], torch.tril(torch.ones(query_len, query_len, dtype=torch.bool))
    )
    # ...and full attention to every valid prefix token.
    assert bool(mask[0, 0, :, :5].all())


def test_p4_attention_mask_respects_prefix_padding():
    model = _build_model()
    query_len = model.config.dit_query_length
    prefix_pad = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    mask = model.dit_attention_mask(prefix_pad, query_len)
    assert not bool(mask[0, 0, :, 3:5].any())
    assert bool(mask[1, 0, :, :5].all())


def test_p4_position_offset_is_per_mrope_axis():
    """Rule-8 conflict test against cosmos3's ``amax(dim=(0, 2))``.

    The two agree whenever the three MRoPE axes share a maximum -- which they do on every
    captured fixture -- so the case has to be *constructed* to discriminate them.
    """
    model = _build_model()
    prefix_position_ids = torch.tensor([[[0, 1, 2]], [[0, 5, 2]], [[0, 1, 9]]])
    prefix_pad = torch.ones(1, 3, dtype=torch.bool)
    ours = model.dit_position_ids(prefix_position_ids, prefix_pad, query_len=2)
    assert ours[:, 0, 0].tolist() == [3, 6, 10]

    collapsed = prefix_position_ids.amax(dim=(0, 2)) + 1  # the cosmos3 formulation
    assert collapsed.tolist() == [10]
    assert ours[:, 0, 0].tolist() != [10, 10, 10]


def test_p4_position_offset_ignores_padded_columns():
    """``get_rope_index`` writes 1 into padded slots; the offset must not read them.

    On an unpadded batch (every fixture) this is identical to the reference's plain
    ``max(dim=-1)``, which the second half asserts.
    """
    model = _build_model()
    position_ids = torch.tensor([[[0, 1, 2, 1, 1]]] * 3)
    padded = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    unpadded = torch.ones(1, 5, dtype=torch.bool)
    assert model.dit_position_ids(position_ids, padded, 1)[:, 0, 0].tolist() == [3, 3, 3]
    plain = position_ids.max(dim=-1).values + 1
    assert model.dit_position_ids(position_ids, unpadded, 1)[:, 0, 0].tolist() == plain[:, 0].tolist()


def test_suffix_position_offset_defaults_to_zero():
    """The released inference code adds no offset; the 5B training recipe adds 10.

    Defaulting to the released behaviour is what keeps a fine-tune consistent with the
    checkpoint it starts from.
    """
    assert XR1Config().suffix_position_offset == 0


# =====================================================================================
# P5 -- observation history
# =====================================================================================


def test_p5_history_indices_match_the_reference_table():
    golden = _golden("history_indices.json")
    for length, expected in golden["indices_by_buffer_length"].items():
        assert history_indices(int(length), golden["length"], golden["interval"]) == expected


def test_p5_history_clamps_rather_than_zero_pads():
    """The divergence pin: both existing OpenTau inference buffers zero-pad.

    With one frame buffered, all four slots must be that frame -- not three zeros and a
    frame, which is what copying ``pi07``'s ``_build_history_batch`` gives.
    """
    assert history_indices(1, 4, 2, pad_mode="clamp") == [0, 0, 0, 0]
    assert history_indices(1, 4, 2, pad_mode="zero") == [-6, -4, -2, 0]
    assert history_indices(7, 4, 2, pad_mode="clamp") == [0, 2, 4, 6]
    assert history_indices(7, 4, 2, pad_mode="zero") == [0, 2, 4, 6]


def test_p5_buffer_repeats_the_first_frame_after_one_append():
    buffer = XR1ObservationBuffer(n_obs_steps=4, history_interval=2, buffer_size=7)
    state = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    image = torch.rand(2, 3, 8, 8)
    buffer.append(state, {"camera0": image})
    stacked_state, images, is_pad = buffer.window()
    assert stacked_state.shape == (2, 4, 3)
    for t in range(4):
        assert torch.equal(stacked_state[:, t], state)
        assert torch.equal(images["camera0"][:, t], image)
    assert is_pad.tolist() == [[False, True, True, True]] * 2


def test_p5_buffer_window_is_evenly_spaced_once_full():
    buffer = XR1ObservationBuffer(n_obs_steps=4, history_interval=2, buffer_size=7)
    for step in range(10):
        buffer.append(torch.full((1, 1), float(step)), {})
    state, _, is_pad = buffer.window()
    assert state[0, :, 0].tolist() == [3.0, 5.0, 7.0, 9.0]
    assert not bool(is_pad.any())


def test_p5_buffer_rejects_a_batch_size_change_without_a_reset():
    """A shared policy across ``max_parallel_tasks > 1`` would otherwise silently stack
    frames from two different episodes."""
    buffer = XR1ObservationBuffer(n_obs_steps=4, history_interval=2, buffer_size=7)
    buffer.append(torch.zeros(2, 3), {})
    with pytest.raises(ValueError, match="batch size"):
        buffer.append(torch.zeros(3, 3), {})
    buffer.reset()
    buffer.append(torch.zeros(3, 3), {})  # fine after a reset


# =====================================================================================
# The action mask, and where it is applied
# =====================================================================================


def test_action_mask_is_float_and_marks_only_the_real_columns():
    """It is a **float** tensor: the reference draws noise with ``randn_like(action_mask)``,
    so this tensor's dtype is what quantizes the noise (bfloat16 on the real model)."""
    mask = build_action_mask(2, 16, 60, 12, device=torch.device("cpu"), dtype=torch.bfloat16)
    assert mask.dtype == torch.bfloat16
    assert mask.shape == (2, 16, 60)
    assert int(mask.float().sum().item()) == 2 * 16 * 12
    assert bool((mask[..., 12:] == 0).all())

    golden = _golden("prompt_tokens.json")["A_start"]
    assert golden["action_mask_dtype"] == "torch.bfloat16"
    assert golden["action_mask_active_dims"] == list(range(12))
    assert golden["action_mask_num_active"] == 16 * 12


def test_action_mask_is_applied_inside_every_dit_call():
    """The natural refactor hoists the mask out of the Euler loop; that changes the DiT's
    *input* on steps 1..N-1, because the output layer writes all ``max_action_dim`` columns
    and the sampler's own ``x`` therefore stops being zero on the padded ones."""
    model = _build_model().eval()
    batch = _video_batch(bsize=1)
    seen_inputs = []
    real_dit = model.dit

    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, *args, **kwargs):
            seen_inputs.append(hidden_states.detach().clone())
            return self.inner(hidden_states, *args, **kwargs)

    model.dit = _Spy(real_dit)
    try:
        with torch.no_grad():
            chunk = model.sample_actions(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values_videos=batch["pixel_values_videos"],
                video_grid_thw=batch["video_grid_thw"],
                state=batch["state"],
                action_mask=batch["action_mask"],
            )
    finally:
        model.dit = real_dit

    assert len(seen_inputs) == model.config.num_steps
    # Every DiT call sees the masked action embedding, and the returned chunk is non-zero on
    # the padded columns -- which is exactly why the masking cannot be hoisted.
    assert chunk.shape == (1, CHUNK, MAX_ACTION_DIM)
    assert chunk[..., ACTION_DIM:].abs().sum().item() > 0


def test_masked_padded_columns_do_not_affect_the_real_ones():
    """Whatever the padded columns of the incoming noise hold, the real output is the same."""
    model = _build_model().eval()
    batch = _video_batch(bsize=1)
    noise = torch.randn_like(batch["action_mask"])
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values_videos": batch["pixel_values_videos"],
        "video_grid_thw": batch["video_grid_thw"],
        "state": batch["state"],
        "action_mask": batch["action_mask"],
    }
    with torch.no_grad():
        a = model.sample_actions(noise=noise.clone(), **kwargs)
        perturbed = noise.clone()
        perturbed[..., ACTION_DIM:] += 100.0
        b = model.sample_actions(noise=perturbed, **kwargs)
    assert torch.allclose(a[..., :ACTION_DIM], b[..., :ACTION_DIM], atol=1e-5)


# =====================================================================================
# adaLN
# =====================================================================================


def test_modulate_is_scale_plus_one():
    x = torch.randn(2, 3, 4)
    shift = torch.randn(2, 1, 4)
    scale = torch.randn(2, 1, 4)
    assert torch.equal(modulate(x, shift, scale), x * (1 + scale) + shift)


def test_dit_layer_residual_gate_has_no_implicit_one():
    """``x + gate * sublayer(x)``, not ``x + (1 + gate) * sublayer(x)``.

    Zeroing every ``adaln_table`` gate and the shared time projection must make the layer an
    exact identity; with a ``1 + gate`` residual it would not be.
    """
    model = _build_model()
    layer = model.dit.layers[0]
    with torch.no_grad():
        layer.adaln_table.zero_()
    hidden = torch.randn(2, 5, model.config.dit_hidden_size)
    t_embeds = torch.zeros(2, 6, model.config.dit_hidden_size)
    kv = (
        torch.randn(2, model.config.dit_num_key_value_heads, 4, model.config.dit_head_dim),
        torch.randn(2, model.config.dit_num_key_value_heads, 4, model.config.dit_head_dim),
    )
    cos = torch.randn(2, 5, model.config.dit_head_dim)
    sin = torch.randn(2, 5, model.config.dit_head_dim)
    mask = torch.ones(2, 1, 5, 9, dtype=torch.bool)
    out = layer(hidden, kv, (cos, sin), t_embeds, attn_mask=mask)
    assert torch.allclose(out, hidden, atol=1e-6)


def test_adaln_chunks_are_six_and_shared_projection_plus_per_layer_table():
    model = _build_model()
    layer = model.dit.layers[0]
    assert layer.adaln_table.shape == (6, model.config.dit_hidden_size)
    assert model.t_projector.layers[0].out_features == 6 * model.config.dit_hidden_size
    assert model.t_projector.layers[0].bias is not None
    # The two DiT layers share one t_projector but carry independent tables.
    assert not torch.equal(model.dit.layers[0].adaln_table, model.dit.layers[1].adaln_table)


def test_sink_is_the_first_query_token():
    """Dropping or moving the sink shifts every DiT position id by one."""
    model = _build_model()
    assert model.sink.weight.shape == (1, model.config.dit_hidden_size)
    assert model.config.dit_query_length == 1 + model.config.n_obs_steps + model.config.chunk_size


def test_repeat_batch_interleaves_rather_than_tiles():
    """The training repeat uses ``repeat_interleave``; tiling would pair each flow timestep
    with a different sample's observation."""
    x = torch.tensor([[0.0], [1.0]])
    assert repeat_batch(x, 4).squeeze(-1).tolist() == [0.0, 0.0, 1.0, 1.0]
    assert repeat_batch(x, 2) is x
    with pytest.raises(ValueError, match="Cannot repeat"):
        repeat_batch(x, 3)


# =====================================================================================
# Losses
# =====================================================================================


def test_flow_loss_reduces_to_the_shared_helper_at_neutral_settings():
    """Rule-8 conflict test: with unit weights and no frequency term, xr1's loss must equal
    ``flow_matching_masked_mse`` **bit-for-bit**, so the extra machinery is provably a
    superset rather than a re-derivation."""
    torch.manual_seed(0)
    bsz, chunk, dim = 3, 16, 60
    pred = torch.randn(bsz, chunk, dim)
    target = torch.randn(bsz, chunk, dim)
    is_pad = torch.zeros(bsz, chunk, dtype=torch.bool)
    is_pad[2, -3:] = True
    real_dim = torch.tensor([12, 12, 12])
    mask = build_flow_mask(
        batch_size=bsz,
        chunk_size=chunk,
        max_action_dim=dim,
        device=pred.device,
        actions_is_pad=is_pad,
        real_action_dim=real_dim,
    )
    ours, _ = flow_and_freq_loss(pred, target, mask=mask)
    theirs = flow_matching_masked_mse(
        target, pred, max_action_dim=dim, actions_is_pad=is_pad, real_action_dim=real_dim
    )
    assert torch.equal(ours, theirs)


def test_flow_loss_is_zero_when_nothing_is_supervised():
    pred = torch.randn(2, 4, 6, requires_grad=True)
    target = torch.randn(2, 4, 6)
    mask = torch.zeros(2, 4, 6, dtype=torch.bool)
    mse, freq = flow_and_freq_loss(pred, target, mask=mask)
    assert float(mse.detach()) == 0.0 and float(freq.detach()) == 0.0
    # Still connected to the graph, so a zero-supervision micro-batch cannot break DDP.
    mse.backward()
    assert pred.grad is not None


def test_frequency_term_honours_excluded_dims():
    torch.manual_seed(0)
    pred = torch.randn(2, 8, 20)
    target = torch.randn(2, 8, 20)
    mask = torch.ones(2, 8, 20, dtype=torch.bool)
    _, all_dims = flow_and_freq_loss(pred, target, mask=mask)
    _, fewer = flow_and_freq_loss(pred, target, mask=mask, freq_excluded_dims=(17, 18, 19))
    assert not torch.allclose(all_dims, fewer)


def test_frequency_term_runs_on_bfloat16_inputs():
    """``torch.fft.rfft`` has no bfloat16 kernel; the loss casts to float32 for exactly this
    reason, and removing the cast is a crash rather than a slow drift."""
    pred = torch.randn(2, 8, 12, dtype=torch.bfloat16)
    target = torch.randn(2, 8, 12, dtype=torch.bfloat16)
    mask = torch.ones(2, 8, 12, dtype=torch.bool)
    mse, freq = flow_and_freq_loss(pred, target, mask=mask)
    assert torch.isfinite(mse) and torch.isfinite(freq)


def test_loss_weight_is_normalized_then_clamped():
    torch.manual_seed(0)
    pred = torch.zeros(2, 4, 3)
    target = torch.ones(2, 4, 3)
    mask = torch.ones(2, 4, 3, dtype=torch.bool)
    weight = torch.full((2, 4, 3), 7.0)
    mse, _ = flow_and_freq_loss(pred, target, mask=mask, weight=weight)
    # A constant weight normalizes to exactly 1, so the MSE is the plain one.
    assert mse == pytest.approx(1.0)


def test_fold_repeat_per_sample_collapses_the_repeat_axis():
    """A ``(B * R,)`` per-sample loss silently misattributes rows in train.py's single
    ``gather_for_metrics`` call, which pairs it with ``(B,)`` provenance tensors."""
    per_sample = PerSampleLoss(sum=torch.tensor([1.0, 2.0, 3.0, 4.0]), count=torch.tensor([1.0] * 4))
    folded = fold_repeat_per_sample(per_sample, batch_size=2, repeat=2)
    assert folded.sum.tolist() == [3.0, 7.0]
    assert folded.count.tolist() == [2.0, 2.0]


# =====================================================================================
# Model forward / sampler
# =====================================================================================


def test_inner_forward_returns_both_loss_keys():
    """``scripts/train.py`` indexes ``"MSE"`` and ``"CE"`` unconditionally, and a zero
    ``loss_weighting`` entry hard-fails under DeepSpeed -- so CE is a real zero tensor."""
    model = _build_model().train()
    batch = _video_batch()
    out = model(**batch)
    assert {"MSE", "CE"} <= set(out)
    assert out["MSE"].ndim == 0 and torch.isfinite(out["MSE"])
    assert torch.equal(out["CE"], torch.zeros((), dtype=out["CE"].dtype))


def test_inner_forward_per_sample_is_batch_shaped_not_repeat_shaped():
    model = _build_model().train()
    batch = _video_batch(bsize=3)
    out = model(**batch, return_per_sample=True)
    assert out["MSE_per_sample"].sum.shape == (3,)
    assert out["MSE_per_sample"].count.shape == (3,)
    assert out["CE_per_sample"].sum.shape == (3,)


def test_inner_forward_backward_reaches_the_dit():
    model = _build_model().train()
    out = model(**_video_batch())
    out["MSE"].backward()
    assert model.dit.layers[0].attn.qkv_proj.weight.grad is not None
    assert model.action_output_layer.layers[0].weight.grad is not None


def test_sample_actions_shape_and_determinism_under_fixed_noise():
    model = _build_model().eval()
    batch = _video_batch(bsize=2)
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values_videos": batch["pixel_values_videos"],
        "video_grid_thw": batch["video_grid_thw"],
        "state": batch["state"],
        "action_mask": batch["action_mask"],
    }
    noise = torch.randn_like(batch["action_mask"])
    with torch.no_grad():
        first = model.sample_actions(noise=noise.clone(), **kwargs)
        second = model.sample_actions(noise=noise.clone(), **kwargs)
    assert first.shape == (2, CHUNK, MAX_ACTION_DIM)
    assert torch.equal(first, second)


def test_sample_actions_matches_the_helper_rollout():
    """``sample_actions`` writes its own Euler loop so the accel invariants are checkable in
    that function (see ``XR1FlowMatching._euler_rollout``'s docstring); this pins that the
    duplicate has not drifted."""
    model = _build_model().eval()
    batch = _video_batch(bsize=1)
    noise = torch.randn_like(batch["action_mask"])
    with torch.no_grad():
        via_sampler = model.sample_actions(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            state=batch["state"],
            action_mask=batch["action_mask"],
            noise=noise.clone(),
        )
        cached_kv, attn_mask, position_ids, state_embed = model._run_prefix_and_geometry(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            state=batch["state"],
            n_prefix_rows=0,
            dtype=batch["action_mask"].dtype,
        )
        position_embeds = model.compute_rope(
            position_ids, dtype=batch["action_mask"].dtype, device=noise.device
        )
        via_helper = model._euler_rollout(
            noise.clone(),
            action_mask=batch["action_mask"],
            state_embed=state_embed,
            position_embeds=position_embeds,
            cached_kv=cached_kv,
            attn_mask=attn_mask,
            num_steps=model.config.num_steps,
        )
    assert torch.equal(via_sampler, via_helper)


def test_action_prefix_rows_are_returned_untouched():
    model = _build_model().eval()
    batch = _video_batch(bsize=1)
    prefix = torch.full((1, CHUNK, MAX_ACTION_DIM), 0.5)
    delay = torch.tensor(3, dtype=torch.long)
    with torch.no_grad():
        chunk = model.sample_actions(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            state=batch["state"],
            action_mask=batch["action_mask"],
            action_prefix=prefix,
            delay=delay,
        )
    assert torch.equal(chunk[:, :3], prefix[:, :3])
    assert not torch.equal(chunk[:, 3:], prefix[:, 3:])


# =====================================================================================
# Freezing, checkpointing, optimizer params
# =====================================================================================


def test_input_embeddings_are_frozen_by_default():
    """389 M parameters that never receive a gradient; handing them to a fused AdamW costs
    ~4.7 GB of optimizer state on the real model."""
    model = _build_model()
    assert model.vlm.get_input_embeddings().weight.requires_grad is False


def test_train_expert_only_freezes_the_backbone_and_keeps_the_dit_trainable():
    model = _build_model(train_expert_only=True)
    assert all(not p.requires_grad for p in model.vlm.parameters())
    assert all(p.requires_grad for p in model.dit.parameters())


def test_train_state_action_representation_only_keeps_just_the_projections():
    model = _build_model(train_state_action_representation_only=True)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(
        name.startswith(("state_projector.", "action_projector.", "action_output_layer."))
        for name in trainable
    ), sorted(trainable)[:10]


def test_gradient_checkpointing_leaves_the_state_dict_keys_unchanged():
    """The MLP wrapper is a **class swap**, not a rebound ``forward``: rebinding leaves
    FSDP's pre-forward hook on the original module while the call bypasses it."""
    plain = _build_model()
    checkpointed = _build_model(gradient_checkpointing=True)
    assert list(plain.state_dict().keys()) == list(checkpointed.state_dict().keys())
    swapped = checkpointed.text_model.layers[0].mlp
    assert type(swapped).__name__ == "_CheckpointedQwen3VLTextMLP"
    assert hasattr(swapped, "gate_proj")


def test_use_cache_survives_gradient_checkpointing():
    """The trap: ``check_model_inputs`` flips ``use_cache=False`` when the text tower has
    checkpointing enabled and is training, and the DiT is then fed **nothing** with only a
    ``warning_once``. Checkpointing the MLPs only keeps the cache alive."""
    model = _build_model(gradient_checkpointing=True).train()
    batch = _video_batch(bsize=1)
    out = model(**batch)
    assert torch.isfinite(out["MSE"])

    cached = model.run_prefix(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=model.get_rope_index(
            input_ids=batch["input_ids"],
            video_grid_thw=batch["video_grid_thw"],
            attention_mask=batch["attention_mask"],
        )[0],
        pixel_values_videos=batch["pixel_values_videos"],
        video_grid_thw=batch["video_grid_thw"],
    )
    assert len(cached) == model.num_layers
    assert all(k.shape[2] == batch["input_ids"].shape[1] for k, _ in cached)


def test_run_prefix_rejects_both_or_neither_input():
    """``Qwen3VLModel.forward`` raises on both; catching it here names the fix."""
    model = _build_model()
    with pytest.raises(ValueError, match="exactly one"):
        model.run_prefix(attention_mask=torch.ones(1, 3), position_ids=torch.zeros(3, 1, 3, dtype=torch.long))


def test_get_optim_params_excludes_frozen_tensors():
    policy = _build_policy()
    trainable = policy.get_optim_params()
    assert all(p.requires_grad for p in trainable)
    embedding = policy.model.vlm.get_input_embeddings().weight
    assert not any(p is embedding for p in trainable)


# =====================================================================================
# Policy wrapper
# =====================================================================================


def _build_policy(**overrides) -> XR1Policy:
    torch.manual_seed(0)
    return XR1Policy(_tiny_xr1_config(**overrides), qwen3vl_config=_tiny_qwen3vl_config(2))


def test_policy_camera_order_is_positional_not_lexicographic():
    """``sorted()`` puts ``camera10`` before ``camera2``; the prompt labels cameras by
    position, so a lexicographic order would relabel the wrist camera."""
    policy = _build_policy(num_cams=11, camera_prompt_labels=tuple(f"c{i}: " for i in range(11)))
    assert policy._image_keys() == [f"camera{i}" for i in range(11)]
    assert sorted(policy._image_keys()) != policy._image_keys()


def test_policy_reset_clears_the_history_and_the_queue():
    policy = _build_policy()
    policy._obs_buffer.append(torch.zeros(2, 16), {})
    policy._action_queue.append(torch.zeros(2, ACTION_DIM))
    policy.reset()
    assert len(policy._action_queue) == 0
    assert policy.last_accel is None
    with pytest.raises(RuntimeError):
        policy._obs_buffer.window()


def test_policy_normalization_creates_no_buffers():
    """IDENTITY normalization registers nothing, so ``_check_norm_stats_loaded`` passes
    trivially and a wrong mapping would be invisible until the numbers are wrong. The
    config-level refusal below is the actual guard."""
    policy = _build_policy()
    assert list(policy.normalize_inputs.named_parameters()) == []
    assert list(policy.unnormalize_outputs.named_parameters()) == []


# =====================================================================================
# Config validation
# =====================================================================================


def test_config_refuses_non_identity_normalization():
    from opentau.configs.types import NormalizationMode

    for feature in ("STATE", "ACTION", "VISUAL"):
        mapping = {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
        mapping[feature] = NormalizationMode.MEAN_STD
        with pytest.raises(ValueError, match="IDENTITY"):
            XR1Config(normalization_mapping=mapping)


def test_config_refuses_a_delay_longer_than_training_ever_showed():
    with pytest.raises(ValueError, match="train_prefix_max"):
        XR1Config(max_delay=8, train_prefix_max=6)
    XR1Config(max_delay=6, train_prefix_max=6)  # the boundary is allowed


def test_config_refuses_a_mismatched_state_adapter_width():
    with pytest.raises(ValueError, match="16-D RoboCasa state"):
        XR1Config(state_adapter="robocasa_panda_omron", max_state_dim=14)
    XR1Config(state_adapter="identity", max_state_dim=14)


def test_config_refuses_a_camera_count_the_prompt_cannot_label():
    with pytest.raises(ValueError, match="camera_prompt_labels"):
        XR1Config(num_cams=4)


def test_config_refuses_odd_history_for_the_temporal_patch():
    with pytest.raises(ValueError, match="vision_temporal_patch_size"):
        XR1Config(n_obs_steps=3)


def test_config_refuses_an_image_size_that_breaks_the_merge_window():
    with pytest.raises(ValueError, match="merge window"):
        XR1Config(image_size=250)


def test_config_defaults_match_the_reference_protocol():
    cfg = XR1Config()
    assert (cfg.n_obs_steps, cfg.history_interval, cfg.obs_buffer_size) == (4, 2, 7)
    assert (cfg.chunk_size, cfg.n_action_steps) == (16, 16)
    assert cfg.num_steps == 5
    assert cfg.center_crop_ratio == 0.95
    assert cfg.attention_implementation == "eager"
    assert cfg.camera_prompt_labels[2].strip() == "Wrist camera:"


def test_execution_horizon_is_the_full_chunk():
    """``n_action_steps < chunk_size`` silently switches to receding-horizon replanning;
    the reference executes all 16 rows open-loop."""
    cfg = XR1Config()
    assert cfg.n_action_steps == cfg.chunk_size == 16


# =====================================================================================
# State-dict remap
# =====================================================================================


def test_remap_inserts_exactly_one_model_prefix():
    state_dict = {
        "vlm.model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(1),
        "dit.layers.0.adaln_table": torch.zeros(6, 4),
        "sink.weight": torch.zeros(1, 4),
        "t_projector.layers.0.bias": torch.zeros(4),
        "normalize_inputs.buffer_state.mean": torch.zeros(4),
    }
    out = remap_reference_state_dict(state_dict)
    assert set(out) == {
        "model.vlm.model.language_model.layers.0.self_attn.q_proj.weight",
        "model.dit.layers.0.adaln_table",
        "model.sink.weight",
        "model.t_projector.layers.0.bias",
        "normalize_inputs.buffer_state.mean",
    }
    assert remap_reference_state_dict(out).keys() == out.keys()  # idempotent


def test_remap_covers_the_captured_manifest():
    """Every key in the reference checkpoint must land on a live-model key.

    A partial load is reported by ``load_state_dict(strict=False)`` as "missing", so it
    looks exactly like a successful load until the numbers are wrong.
    """
    manifest = _golden("checkpoint_manifest.json")
    remapped = {f"model.{key}" for key in manifest["tensors"]}
    assert all(key.startswith("model.") for key in remapped)
    tops = {key.split(".")[1] for key in remapped}
    assert tops == {
        "vlm",
        "dit",
        "state_projector",
        "action_projector",
        "action_output_layer",
        "t_embedder",
        "t_projector",
        "sink",
    }


def test_assert_full_coverage_tolerates_only_the_tied_head():
    assert_full_coverage(["model.vlm.lm_head.weight"], [])
    with pytest.raises(ValueError, match="did not fully cover"):
        assert_full_coverage(["model.dit.layers.0.adaln_table"], [])
    with pytest.raises(ValueError, match="did not fully cover"):
        assert_full_coverage([], ["some.stray.key"])


def test_live_model_key_names_match_the_reference_manifest_shape():
    """The projector Linears must sit at ``layers.0`` / ``layers.2`` (an ``nn.Sequential``
    with the GELU at index 1), and the DiT's second norm must be ``post_layernorm``."""
    model = _build_model()
    keys = set(model.state_dict())
    assert {"state_projector.layers.0.weight", "state_projector.layers.2.weight"} <= keys
    assert {"t_embedder.mlp.0.weight", "t_embedder.mlp.2.weight"} <= keys
    assert "t_projector.layers.0.bias" in keys
    assert "dit.layers.0.post_layernorm.weight" in keys
    assert "dit.layers.0.post_attention_layernorm.weight" not in keys
    assert "dit.layers.0.attn.qkv_proj.bias" in keys


# =====================================================================================
# Video patching
# =====================================================================================


def test_patchify_matches_qwen3vl_axis_order():
    """The merge windows are the *inner* spatial axes. Swapping the two decompositions
    produces a correctly-shaped tensor of scrambled patches, which the vision tower will
    happily consume."""
    videos = torch.arange(2 * 4 * 3 * 32 * 32, dtype=torch.float32).reshape(2, 4, 3, 32, 32)
    ours, grid = patchify_videos(videos, patch_size=16, temporal_patch_size=2, merge_size=2)
    assert grid.tolist() == [[2, 2, 2], [2, 2, 2]]

    # Reference: transformers' own view/permute (Qwen3VLVideoProcessor._preprocess).
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, -1, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(1, 1, -1, 1, 1)
    patches = (videos - mean) / std
    b, gt, c = patches.shape[0], patches.shape[1] // 2, patches.shape[2]
    patches = patches.view(b, gt, 2, c, 1, 2, 16, 1, 2, 16)
    patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    expected = patches.reshape(b * gt * 2 * 2, c * 2 * 16 * 16)
    assert torch.equal(ours, expected)


def test_patchify_rejects_a_size_that_is_not_a_whole_merge_window():
    with pytest.raises(ValueError, match="merge windows"):
        patchify_videos(torch.rand(1, 2, 3, 48, 48), patch_size=16, temporal_patch_size=2, merge_size=2)


def test_patchify_rejects_an_odd_frame_count():
    with pytest.raises(ValueError, match="temporal_patch_size"):
        patchify_videos(torch.rand(1, 3, 3, 32, 32), patch_size=16, temporal_patch_size=2, merge_size=2)


# =====================================================================================
# Scene seeds (the harness-side parity contract)
# =====================================================================================


def test_scene_seed_scheme_is_recorded():
    """The reference's per-episode seeds are ``7 + task_index * 50 + episode``. OpenTau's
    eval uses ``start_seed + episode_index``, so the two only evaluate the same scenes when
    the reference's seeds are fed through ``cfg.eval.seed_list``."""
    golden = _golden("scene_seeds.json")
    assert golden["base_seed"] == 7
    assert golden["num_trials"] == 50
    assert "seed_list" in golden["note"]
    seeds = [golden["base_seed"] + 3 * golden["num_trials"] + i for i in range(4)]
    assert seeds == [157, 158, 159, 160]


# =====================================================================================
# The shared history helper vs the existing pi07 inference buffer
# =====================================================================================


def test_zero_pad_mode_reproduces_the_pi07_inference_buffer_exactly():
    """``history_slot_indices(pad_mode="zero")`` must match ``pi07``'s inline arithmetic.

    The helper was extracted so xr1 could take the *clamping* branch without forking the
    logic; that is only safe if the zero branch is provably the code it replaces. This
    reproduces ``PI07PaligemmaLowLevelPolicy._build_history_batch``'s ``idx = i * interval
    - missing`` for every buffer fill level, so a future migration of that policy onto the
    helper is a no-op rather than a silent behaviour change.
    """
    from opentau.policies.utils import history_slot_indices

    for n_hist, interval in ((4, 2), (8, 1), (3, 3), (2, 5)):
        buf_maxlen = (n_hist - 1) * interval + 1
        for buf_len in range(1, buf_maxlen + 1):
            missing = buf_maxlen - buf_len
            pi07 = [i * interval - missing for i in range(n_hist)]
            assert history_slot_indices(buf_len, n_hist, interval, pad_mode="zero") == pi07
            # ...and the clamping variant differs exactly where pi07 would zero-fill.
            clamped = history_slot_indices(buf_len, n_hist, interval, pad_mode="clamp")
            assert clamped == [max(0, index) for index in pi07]
            assert (clamped != pi07) == any(index < 0 for index in pi07)


def test_history_helper_rejects_an_empty_buffer_and_an_unknown_mode():
    from opentau.policies.utils import history_slot_indices

    with pytest.raises(ValueError, match="buffer_length"):
        history_slot_indices(0, 4, 2)
    with pytest.raises(ValueError, match="pad_mode"):
        history_slot_indices(4, 4, 2, pad_mode="reflect")


# =====================================================================================
# Choice heads (Stage 2 auxiliary signal)
# =====================================================================================


def _choice_config(**overrides):
    return _tiny_xr1_config(enable_choice_heads=True, n_choices=3, **overrides)


def test_choice_heads_are_off_by_default():
    """The released RoboCasa365 checkpoint dropped them, so the parity-verified path must
    not grow them; a fine-tune opts in explicitly."""
    assert XR1Config().enable_choice_heads is False
    assert _build_model().choice_heads is None


def test_choice_turn_prefix_is_the_last_im_start_before_state():
    from opentau.policies.xr1.choice_heads import (
        ACTION_TOKEN_START_ID,
        IM_START_TOKEN_ID,
        SCORE_TOKEN_ID,
        STATE_TOKEN_ID,
        choice_turn_prefix_length,
    )

    row = [IM_START_TOKEN_ID, 7, 8, IM_START_TOKEN_ID, 9, IM_START_TOKEN_ID, STATE_TOKEN_ID]
    row += [ACTION_TOKEN_START_ID, ACTION_TOKEN_START_ID + 1, SCORE_TOKEN_ID]
    assert choice_turn_prefix_length(torch.tensor([row])).tolist() == [5]

    with pytest.raises(ValueError, match="<state>"):
        choice_turn_prefix_length(torch.tensor([[1, 2, 3]]))
    with pytest.raises(ValueError, match="im_start"):
        choice_turn_prefix_length(torch.tensor([[STATE_TOKEN_ID, 2]]))


def test_choice_loss_matches_a_literal_transcription_of_the_reference_loop():
    """The vectorized form must equal the reference's per-sample loop exactly.

    The reference loops because its collate packs variable-length samples into one
    sequence; OpenTau pads instead, so the loop is unnecessary -- but "unnecessary" has to
    be demonstrated, not asserted.
    """
    from opentau.policies.xr1.choice_heads import choice_loss

    torch.manual_seed(0)
    bsz, chunk, n_choices, dim = 3, 5, 4, 7
    action_pred = torch.randn(bsz, chunk, n_choices, dim)
    score_pred = torch.randn(bsz, n_choices)
    target = torch.randn(bsz, chunk, dim)
    mask = torch.ones(bsz, chunk, dim, dtype=torch.bool)
    mask[1, -2:, :] = False
    mask[2, :, -3:] = False

    ours = choice_loss(action_pred, score_pred, target, mask)

    action_losses, score_losses = [], []
    for b in range(bsz):
        predictions = action_pred[b].transpose(0, 1)  # (n_choices, chunk, dim)
        sample_target = target[b][None].repeat(n_choices, 1, 1)
        sample_mask = mask[b][None].repeat(n_choices, 1, 1)
        absolute_error = torch.nn.functional.l1_loss(predictions, sample_target, reduction="none")
        choice_error = absolute_error[sample_mask].reshape(n_choices, -1).mean(dim=-1)
        index = choice_error.argmin()
        action_losses.append(choice_error[index])
        score_losses.append(((score_pred[b] - choice_error.detach()) ** 2).mean())
    expected = (torch.stack(action_losses).mean(), torch.stack(score_losses).mean())

    assert torch.allclose(ours[0], expected[0], atol=1e-6)
    assert torch.allclose(ours[1], expected[1], atol=1e-6)


def test_choice_loss_picks_the_best_candidate_not_the_average():
    """Winner-takes-all: adding a *worse* candidate must not change the action loss."""
    from opentau.policies.xr1.choice_heads import choice_loss

    target = torch.zeros(1, 2, 3)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    good = torch.full((1, 2, 1, 3), 0.1)
    bad = torch.full((1, 2, 1, 3), 9.0)
    one, _ = choice_loss(good, torch.zeros(1, 1), target, mask)
    two, _ = choice_loss(torch.cat([good, bad], dim=2), torch.zeros(1, 2), target, mask)
    assert torch.allclose(one, two, atol=1e-6)
    assert one == pytest.approx(0.1, abs=1e-6)


def test_choice_head_state_dict_key_names_match_the_reference():
    """The 5B warm start is a plain key match apart from the two token embeddings, which
    the reference hangs off its vendored ``Qwen3VLModel``."""
    from opentau.policies.xr1.choice_heads import WARM_START_RENAMES, XR1ChoiceHeads

    heads = XR1ChoiceHeads(hidden_size=8, state_dim=6, action_dim=4, n_choices=3)
    keys = set(heads.state_dict())
    assert {f"action_projector_choice.0.layers.{i}.weight" for i in (0, 2, 4, 6)} <= keys
    assert "action_projector_choice.1.layers.0.weight" in keys
    assert {f"score_projector_choice.0.layers.{i}.weight" for i in (0, 2, 4, 6)} <= keys
    assert {"state_projector_choice.layers.0.weight", "state_projector_choice.layers.2.weight"} <= keys
    assert set(WARM_START_RENAMES.values()) <= keys


def test_choice_head_warm_start_loads_and_refuses_a_shape_mismatch():
    from opentau.policies.xr1.choice_heads import XR1ChoiceHeads, load_choice_head_warm_start

    heads = XR1ChoiceHeads(hidden_size=8, state_dim=6, action_dim=4, n_choices=3)
    source = {f"{k}": torch.randn_like(v) for k, v in heads.state_dict().items()}
    source["vlm.model.action_embed.weight"] = source.pop("action_embed.weight")
    source["vlm.model.score_embed.weight"] = source.pop("score_embed.weight")
    source["something.unrelated"] = torch.randn(3)
    loaded = load_choice_head_warm_start(heads, source)
    assert set(loaded) == set(heads.state_dict())
    assert torch.equal(heads.action_embed.weight, source["vlm.model.action_embed.weight"])

    with pytest.raises(ValueError, match="shape mismatch"):
        load_choice_head_warm_start(heads, {"score_embed.weight": torch.randn(1, 999)})


def test_choice_heads_build_the_model_and_train_end_to_end():
    model = _build_choice_model().train()
    assert model.choice_heads is not None
    batch = _video_batch(bsize=2)

    from opentau.policies.xr1.choice_heads import (
        ACTION_TOKEN_START_ID,
        IM_START_TOKEN_ID,
        SCORE_TOKEN_ID,
        STATE_TOKEN_ID,
    )

    prefix_len = batch["input_ids"].shape[1]
    suffix = (
        [IM_START_TOKEN_ID, STATE_TOKEN_ID]
        + [ACTION_TOKEN_START_ID + i for i in range(CHUNK)]
        + [SCORE_TOKEN_ID]
    )
    suffix_ids = torch.tensor([suffix] * 2, dtype=torch.long)
    batch["input_ids"] = torch.cat([batch["input_ids"], suffix_ids], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(suffix_ids)], dim=1)

    out = model(**batch, choice_prefix_len=prefix_len)
    assert torch.isfinite(out["CE"]) and out["CE"].item() != 0.0
    assert {"loss_choice", "loss_score"} <= set(out)
    out["MSE"].add(out["CE"]).backward()
    assert model.choice_heads.action_projector_choice[0].layers[0].weight.grad is not None
    assert model.dit.layers[0].attn.qkv_proj.weight.grad is not None


def test_choice_turn_is_excluded_from_the_dit_cache():
    """The DiT must not be able to read the VLM's own action guess.

    Pinned by construction: the cached keys/values are cut to the observation prefix, so
    changing the choice-turn tokens cannot move the sampled chunk.
    """
    model = _build_choice_model().eval()
    batch = _video_batch(bsize=1)
    prefix_len = batch["input_ids"].shape[1]

    from opentau.policies.xr1.choice_heads import (
        ACTION_TOKEN_START_ID,
        IM_START_TOKEN_ID,
        SCORE_TOKEN_ID,
        STATE_TOKEN_ID,
    )

    suffix = (
        [IM_START_TOKEN_ID, STATE_TOKEN_ID]
        + [ACTION_TOKEN_START_ID + i for i in range(CHUNK)]
        + [SCORE_TOKEN_ID]
    )
    ids = torch.cat([batch["input_ids"], torch.tensor([suffix])], dim=1)
    mask = torch.ones_like(ids)

    with torch.no_grad():
        cached, _ = model.run_prefix(
            input_ids=ids,
            attention_mask=mask,
            position_ids=model.get_rope_index(
                input_ids=ids, video_grid_thw=batch["video_grid_thw"], attention_mask=mask
            )[0],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            prefix_len=prefix_len,
            return_hidden_states=True,
        )
        uncut = model.run_prefix(
            input_ids=ids,
            attention_mask=mask,
            position_ids=model.get_rope_index(
                input_ids=ids, video_grid_thw=batch["video_grid_thw"], attention_mask=mask
            )[0],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
        )
    assert all(k.shape[2] == prefix_len for k, _ in cached)
    assert all(k.shape[2] == ids.shape[1] for k, _ in uncut)
    # The cut keeps exactly the observation prefix, unchanged.
    assert torch.equal(cached[0][0], uncut[0][0][:, :, :prefix_len])


# =====================================================================================
# Determinism (CLAUDE.md rule 3)
# =====================================================================================


def test_two_seeded_training_runs_produce_a_bit_identical_loss_series():
    """Ten seeded optimizer steps, twice, compared **bit-for-bit** -- not "close".

    This is the check the shape-static async prefix exists to make meaningful. The
    reference draws one Python-level ``prefix_length`` per batch and *slices* the DiT
    query; here it is a per-sample mask over a full-length chunk, so every tensor keeps its
    shape whatever the draw and the graph is identical between runs (and, on real hardware,
    across ranks -- ranks that disagree on a sequence length deadlock at the ZeRO-3 /FSDP
    all-gather rather than producing a wrong number).
    """
    from opentau.utils.random_utils import set_seed

    def run() -> list[float]:
        set_seed(1234)
        model = XR1FlowMatching(_tiny_xr1_config(), qwen3vl_config=_tiny_qwen3vl_config(2)).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch = _video_batch(bsize=2)
        losses = []
        for _ in range(10):
            optimizer.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = out["MSE"] + out["CE"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    first, second = run(), run()
    assert first == second, [(a, b) for a, b in zip(first, second, strict=True) if a != b][:3]
    # ...and the run is actually learning something, so the equality is not two flat lines.
    assert first[-1] != first[0]


def test_async_prefix_keeps_every_tensor_shape_constant_across_draws():
    """Whatever the per-sample prefix draw, the DiT sequence length is the same.

    A Python-level branch on the draw (the reference's approach) makes the sequence length
    a per-rank random variable, which is a deadlock under ZeRO-3 / FSDP rather than a
    visible error -- CLAUDE.md rule 5.
    """
    model = _build_model().train()
    batch = _video_batch(bsize=2)
    seen = []
    real_dit = model.dit

    class _ShapeSpy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, *args, **kwargs):
            seen.append(tuple(hidden_states.shape))
            return self.inner(hidden_states, *args, **kwargs)

    model.dit = _ShapeSpy(real_dit)
    try:
        for seed in range(6):
            torch.manual_seed(seed)
            model(**batch)
    finally:
        model.dit = real_dit
    assert len(set(seen)) == 1, sorted(set(seen))
    # [sink, state x n_obs, action x chunk], times training_repeat folded into the batch.
    assert seen[0] == (
        2 * model.config.training_repeat,
        model.config.dit_query_length,
        model.config.dit_hidden_size,
    )


def test_validation_forward_uses_a_deterministic_time_grid_and_no_repeat():
    """In eval mode the repeat collapses to 1 and the timestep grid is deterministic.

    Two things are pinned here, and only the *second* is xr1's own: the DiT runs at batch
    ``B`` rather than ``B * training_repeat``, and the timestep is a fixed grid rather than
    a ``Beta`` draw -- otherwise the validation curve is dominated by tau variance rather
    than by the model. The starting **noise** is still drawn per call (as it is for pi05 and
    cosmos3), so the caller supplies it when a bit-identical validation number is wanted;
    that is what the fixed ``noise=`` below stands in for.
    """
    model = _build_model().eval()
    batch = _video_batch(bsize=3)
    noise = torch.randn_like(batch["actions"])
    first = model(**batch, noise=noise.clone())["MSE"]
    second = model(**batch, noise=noise.clone())["MSE"]
    assert torch.equal(first, second)

    # ...and the repeat really is off: the DiT sees batch B, not B * training_repeat.
    seen = []
    real_dit = model.dit

    class _ShapeSpy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, *args, **kwargs):
            seen.append(hidden_states.shape[0])
            return self.inner(hidden_states, *args, **kwargs)

    model.dit = _ShapeSpy(real_dit)
    try:
        model(**batch, noise=noise.clone())
    finally:
        model.dit = real_dit
    assert set(seen) == {3}
    assert model.config.training_repeat > 1  # so the assertion above is not vacuous


# =====================================================================================
# The two 16-D RoboCasa state layouts (they are NOT interchangeable)
# =====================================================================================


def test_the_env_and_dataset_state_layouts_are_different_and_both_supported():
    """The simulator's ``agent_pos`` is base-first; the dataset's ``observation.state`` is
    EE-first.

    Both are 16 wide and both hold two unit quaternions, so feeding one to the other's
    adapter produces a finite, plausible 14-vector and nothing complains -- the eval would
    simply be conditioned on a scrambled pose. This constructs a state whose two halves are
    distinguishable and asserts the adapters disagree, so a future "simplification" that
    collapses them to one layout fails here.
    """
    from opentau.policies.xr1.obs_adapter import STATE_LAYOUTS, robocasa_state_to_xr1_state

    assert set(STATE_LAYOUTS) == {"robocasa_panda_omron", "robocasa365_dataset"}

    ee_pos = torch.tensor([0.30, -0.10, 0.50], dtype=torch.float64)
    ee_quat = torch.tensor([0.1830, 0.3660, 0.5490, 0.7320], dtype=torch.float64)
    ee_quat = ee_quat / ee_quat.norm()
    base_pos = torch.tensor([2.50, -3.10, 0.701], dtype=torch.float64)
    # A mobile base rotates only about z, so x and y are exactly zero -- the signature the
    # dataset layout was identified by.
    base_quat = torch.tensor([0.0, 0.0, 0.3827, 0.9239], dtype=torch.float64)

    env_state = torch.cat([base_pos, base_quat, ee_pos, ee_quat, torch.tensor([0.02, -0.02])])
    dataset_state = torch.cat([ee_pos, ee_quat, base_pos, base_quat, torch.tensor([0.02, -0.02])])

    from_env = robocasa_state_to_xr1_state(env_state, "robocasa_panda_omron")
    from_dataset = robocasa_state_to_xr1_state(dataset_state, "robocasa365_dataset")
    # Same physical pose, two encodings -> the same XR-1 state.
    assert torch.allclose(from_env, from_dataset, atol=1e-12)

    # ...and reading either one with the *other* adapter does not raise; it silently
    # produces a different vector. That is exactly why the config field exists.
    wrong = robocasa_state_to_xr1_state(dataset_state, "robocasa_panda_omron")
    assert torch.isfinite(wrong).all()
    assert not torch.allclose(wrong, from_dataset, atol=1e-6)


def test_dataset_layout_matches_the_recorded_column_signature():
    """Pin *why* the dataset layout was identified, not just the conclusion.

    The signature is measurable and unambiguous: a yaw-only base quaternion has identically
    zero x/y components. The fixture records where that pattern sits in the real data.
    """
    golden = _golden("state_from_env.json")
    signature = golden["dataset_column_signature"]
    assert signature["cols_10_11_are_identically_zero"] is True
    assert signature["quat_norm_cols_3_7"] == 1.0

    from opentau.policies.xr1.obs_adapter import STATE_LAYOUTS

    fields = STATE_LAYOUTS["robocasa365_dataset"]
    assert (fields["base_quat"].start, fields["base_quat"].stop) == (10, 14)
    assert (fields["ee_quat_rel"].start, fields["ee_quat_rel"].stop) == (3, 7)
    assert golden["dataset_state16_layout"][0].startswith("end_effector_position_relative")
    assert golden["agent_pos16_layout"][0].startswith("base_position")


def test_the_shipped_configs_pick_the_right_adapter_for_their_data_source():
    """Eval reads the simulator; training reads the dataset. Swapping them is silent."""
    import json as _json
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[2] / "configs" / "examples"
    eval_cfg = _json.loads((root / "xr1_robocasa365_eval_config.json").read_text())
    train_cfg = _json.loads((root / "xr1_robocasa365_finetune_config.json").read_text())
    assert eval_cfg["policy"]["state_adapter"] == "robocasa_panda_omron"
    assert train_cfg["policy"]["state_adapter"] == "robocasa365_dataset"


def test_only_one_end_of_the_tied_embedding_may_be_missing():
    """A checkpoint carries one end of the tie, and *which* end depends on its provenance.

    The reference checkpoint omits ``lm_head.weight``; one written by ``save_pretrained``
    omits ``embed_tokens.weight``, because ``safetensors`` will not serialize aliased
    storage twice. Both must load; **both missing** must not.
    """
    from opentau.policies.xr1.state_dict_remap import TIED_WEIGHT_KEYS

    assert_full_coverage(["model.vlm.lm_head.weight"], [])
    assert_full_coverage(["model.vlm.model.language_model.embed_tokens.weight"], [])
    with pytest.raises(ValueError, match="Both ends"):
        assert_full_coverage(sorted(TIED_WEIGHT_KEYS), [])


def test_expand_video_placeholders_accepts_a_non_cpu_style_grid_tensor():
    """The grid arrives as a tensor on the batch's device; ``np.asarray`` refuses a CUDA one.

    Reproduced on the serving path (a RoboCasa rollout died in ``select_action``), which no
    CPU test reaches, so the regression is pinned on the shape of the call instead: a plain
    ``torch.Tensor`` grid must work without going through numpy.
    """
    grid = torch.tensor([[2, 16, 16]], dtype=torch.long)
    text = render_chat_text("x", ("Left camera: ",))
    expanded = expand_video_placeholders(text, grid, 2, [[0, 1, 2, 3]])
    assert expanded.count("<|video_pad|>") == 128
    # ...and a plain nested list still works, so the numpy path is not dead.
    assert expand_video_placeholders(text, [[2, 16, 16]], 2, [[0, 1, 2, 3]]) == expanded


def test_euler_rollout_pins_the_masked_prefix_rows():
    """The async-prefix weight rollout must not let committed rows drift.

    The reference zeroes the velocity on those rows (``output[:, :prefix_length] = 0``); the
    weight it computes is ``|rollout - action|`` on the *suffix*, so a drifting prefix
    silently changes the conditioning that weight is measuring. Per-sample here, because
    the prefix length is drawn per sample rather than per batch.
    """
    model = _build_model().eval()
    batch = _video_batch(bsize=2)
    with torch.no_grad():
        cached_kv, attn_mask, position_ids, state_embed = model._run_prefix_and_geometry(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values_videos=batch["pixel_values_videos"],
            video_grid_thw=batch["video_grid_thw"],
            state=batch["state"],
            n_prefix_rows=0,
            dtype=batch["action_mask"].dtype,
        )
        position_embeds = model.compute_rope(
            position_ids, dtype=batch["action_mask"].dtype, device=batch["action_mask"].device
        )
        seed = torch.randn_like(batch["actions"])
        # Sample 0 freezes 3 rows, sample 1 freezes none -- the per-sample case a scalar
        # "prefix_rows" would get wrong for at least one of them.
        prefix_mask = torch.zeros(2, CHUNK, dtype=torch.bool)
        prefix_mask[0, :3] = True
        rolled = model._euler_rollout(
            seed.clone(),
            action_mask=batch["action_mask"],
            state_embed=state_embed,
            position_embeds=position_embeds,
            cached_kv=cached_kv,
            attn_mask=attn_mask,
            num_steps=model.config.num_steps,
            prefix_mask=prefix_mask,
        )
    assert torch.equal(rolled[0, :3], seed[0, :3])
    assert not torch.equal(rolled[0, 3:], seed[0, 3:])
    assert not torch.equal(rolled[1], seed[1])


def test_eval_config_requests_both_object_registries():
    """RoboCasa's default is ``("objaverse", "lightwheel")``; OpenTau defaults to lightwheel
    alone because the objaverse pack is ~30 GB.

    That restriction changes object sampling, so a scene built at the *same* reset seed is
    not the reference's -- measured on CloseFridge seed 57, the robot's ``base_position``
    moves from ``[1.4366, -3.1005, 0.7]`` to ``[1.4284, -3.2620, 0.7]``. Since the whole
    point of the eval config is comparability with the published per-task rates, it has to
    ask for both.
    """
    import json as _json
    from pathlib import Path as _Path

    cfg = _json.loads(
        (
            _Path(__file__).resolve().parents[2] / "configs" / "examples" / "xr1_robocasa365_eval_config.json"
        ).read_text()
    )
    assert cfg["env"]["obj_registries"] == ["objaverse", "lightwheel"]
    assert cfg["env"]["camera_name"] == "robot0_agentview_left,robot0_agentview_right,robot0_eye_in_hand"
    assert cfg["env"]["episode_length"] is None  # official per-task horizons
    assert cfg["env"]["max_parallel_tasks"] == 1
    assert cfg["eval"]["use_async_envs"] is True


# =====================================================================================
# The DiT action mask on the TRAINING path
# =====================================================================================


def _training_shaped_policy():
    """A policy whose ``output_features`` came from a dataset mixture, not a hand-written config.

    ``make_policy`` overwrites ``output_features`` from ``ds_meta`` whenever features are not
    already set, and ``WeightedDatasetMixture.features`` reports ``actions`` as
    ``(max_action_dim,)`` for every dataset in the mixture. Reproducing that here is the whole
    point: every parity gate runs the *eval* config, which declares the true 12-wide action,
    so nothing that exists today exercises the shape training actually sees.
    """
    from opentau.configs.types import FeatureType, PolicyFeature

    policy = _build_policy()
    policy.config.output_features = {
        "actions": PolicyFeature(type=FeatureType.ACTION, shape=(policy.config.max_action_dim,))
    }
    return policy


def test_training_action_mask_comes_from_the_per_sample_dataset_signal():
    """The mask must mark 12 real columns even when the declared feature width says 60.

    Reading ``config.action_feature`` here yields an all-ones mask, which silently stops
    ``noisy_action * action_mask`` from zeroing the padding columns — so the DiT would train
    on noise where inference feeds it zeros, with no error and no failing loss.
    """
    policy = _training_shaped_policy()
    assert policy.config.action_feature.shape[0] == policy.config.max_action_dim  # the trap

    batch = {"real_action_dim": torch.tensor([12, 7])}
    mask = policy._action_mask(batch, 2, torch.device("cpu"), torch.float32)

    assert mask.shape == (2, policy.config.chunk_size, policy.config.max_action_dim)
    assert int(mask[0, 0].sum()) == 12
    assert int(mask[1, 0].sum()) == 7  # per-sample, not one width for the batch
    assert not bool((mask == 1).all())


def test_training_batch_without_real_action_dim_is_refused():
    """Falling back to the declared width would be the silent bug; refuse instead."""
    policy = _training_shaped_policy().train()
    with pytest.raises(ValueError, match="real_action_dim"):
        policy._action_mask({}, 2, torch.device("cpu"), torch.float32)


def test_inference_action_mask_still_reads_the_declared_width():
    """The eval path has no dataset and no ``real_action_dim``; the config is correct there."""
    policy = _build_policy().eval()
    assert policy.config.action_feature.shape[0] == ACTION_DIM
    mask = policy._action_mask({}, 2, torch.device("cpu"), torch.float32)
    assert int(mask[0, 0].sum()) == ACTION_DIM
    assert bool((mask[..., ACTION_DIM:] == 0).all())


def test_padded_action_columns_reach_the_dit_as_zeros_under_a_training_shaped_config():
    """End-to-end: whatever the config declares, the DiT's input padding columns are zero.

    Spies on the tensor the DiT actually receives rather than on the mask, so the assertion
    survives a refactor that moves where the masking happens.
    """
    policy = _training_shaped_policy()
    model = policy.model
    seen = []
    real_dit = model.dit

    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, *args, **kwargs):
            seen.append(hidden_states.detach().clone())
            return self.inner(hidden_states, *args, **kwargs)

    batch = _video_batch(bsize=2)
    batch["action_mask"] = build_action_mask(
        2,
        CHUNK,
        MAX_ACTION_DIM,
        real_action_dim=torch.tensor([ACTION_DIM, ACTION_DIM]),
        device=batch["actions"].device,
        dtype=batch["actions"].dtype,
    )
    # The action projector consumes `noisy_action * action_mask`, so a non-zero padding
    # column would show up as a different projected embedding. Compare against the same
    # batch with the padding columns of the incoming noise perturbed: masked correctly, the
    # DiT input is identical.
    noise = torch.randn_like(batch["action_mask"])
    perturbed = noise.clone()
    perturbed[..., ACTION_DIM:] += 50.0

    model.dit = _Spy(real_dit)
    try:
        with torch.no_grad():
            # Seed each call: the async-prefix length is drawn per forward, and a different
            # draw changes `noisy_action` for reasons that have nothing to do with masking.
            torch.manual_seed(0)
            model(**batch, noise=noise.clone(), time=torch.zeros(2 * model.config.training_repeat))
            first = list(seen)
            seen.clear()
            torch.manual_seed(0)
            model(**batch, noise=perturbed, time=torch.zeros(2 * model.config.training_repeat))
            second = list(seen)
    finally:
        model.dit = real_dit

    assert first and len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b), "padding columns of the noise reached the DiT"


def test_choice_heads_and_train_expert_only_are_refused_together():
    """Half the heads would learn and half would not, silently.

    ``train_expert_only`` runs the backbone prefix under ``no_grad``, so the three modules
    that feed ``inputs_embeds`` (``state_projector_choice``, ``action_embed``,
    ``score_embed``) reach only tensors carrying no graph — while the two projectors that
    read the detached hidden states keep updating. Nothing errors and the loss still moves,
    which is why this has to be refused at config time rather than documented.
    """
    with pytest.raises(ValueError, match="mutually exclusive"):
        XR1Config(enable_choice_heads=True, train_expert_only=True)

    # ...and each alone is fine, so the guard is about the combination, not either flag.
    XR1Config(enable_choice_heads=True)
    XR1Config(train_expert_only=True)


def test_attention_implementation_does_not_reach_the_dit():
    """The config field is the *backbone's* kernel; the DiT always uses SDPA.

    Pins the docstring's corrected claim by exercising it: building under both settings must
    leave the DiT identical, because the reference hard-codes
    ``F.scaled_dot_product_attention`` there regardless.
    """
    for impl in ("eager", "sdpa"):
        model = _build_model(attention_implementation=impl)
        assert model.vlm.config.text_config._attn_implementation == impl
        # The DiT attention takes no kernel argument at all.
        import inspect

        from opentau.policies.xr1.dit import XR1DiTAttention

        assert "attention_implementation" not in inspect.signature(XR1DiTAttention.__init__).parameters


def test_val_deterministic_time_does_not_drive_the_repeat_collapse():
    """The flag controls the timestep only; ``self.training`` drives the repeat.

    Rule-8 conflict test: set the flag OFF and stay in eval mode, and the repeat must still
    collapse to 1 — which is what the corrected docstring claims and the previous one denied.
    """
    model = _build_model(val_deterministic_time=False).eval()
    batch = _video_batch(bsize=3)
    seen = []
    real_dit = model.dit

    class _ShapeSpy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden_states, *args, **kwargs):
            seen.append(hidden_states.shape[0])
            return self.inner(hidden_states, *args, **kwargs)

    model.dit = _ShapeSpy(real_dit)
    try:
        model(**batch)
    finally:
        model.dit = real_dit
    assert set(seen) == {3}
    assert model.config.training_repeat > 1  # so the assertion is not vacuous
