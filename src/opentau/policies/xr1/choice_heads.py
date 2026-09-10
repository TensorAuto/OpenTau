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

"""VLM-side choice-policy heads (Xiaomi-Robotics-1's auxiliary training signal).

Alongside the DiT's flow loss, the reference trains the **VLM** to propose ``n_choices``
candidate action chunks and to score them, through three heads reading the hidden states of
three special-token families:

* ``<state>`` (id 151670) -- one token; its embedding is *replaced* by a projection of the
  proprioceptive state, so the language model is conditioned on the robot's pose.
* ``<a_0> .. <a_59>`` (ids 151671-151730) -- **one token per action timestep** (not per
  action dimension; the reference's chunk is at most 60 steps). Each token's hidden state
  is projected to ``n_choices x action_dim`` candidate actions for that timestep.
* ``<score>`` (id 151669) -- one token; projected to ``n_choices`` predicted errors.

The loss is winner-takes-all: the candidate with the lowest masked L1 against the ground
truth supervises the action head, and its (detached) error supervises the score head. So
the VLM learns both to propose and to rank, and the DiT keeps its own flow objective.

**Two things make this correct rather than merely present:**

1. **The choice tokens must be excluded from the DiT's KV cache.** The choice turn is
   appended *after* the observation prompt, and the reference truncates the cache to the
   observation prefix before the DiT reads it (``_unpad`` /
   ``action_vlm_condition_segments``). Without the truncation the DiT can simply copy the
   VLM's own action guess, and the reference's paper attributes a real regression to that.
   :func:`choice_turn_prefix_length` computes the cut, and it is the last ``<|im_start|>``
   before ``<state>`` -- exactly the reference's rule.
2. **``Qwen3VLModel.forward`` takes exactly one of ``input_ids`` / ``inputs_embeds``.**
   Passing both raises. So the injection builds ``inputs_embeds`` from ``input_ids``,
   ``masked_scatter``s the three sources in, and leaves the ``<|video_pad|>`` rows
   **untouched** -- with ``input_ids=None`` the backbone falls back to comparing embeddings
   against ``embed(video_token_id)`` to find the video slots, which still resolves exactly
   because those rows were not modified.

The heads are **off by default** (``XR1Config.enable_choice_heads``): the released
RoboCasa365 checkpoint dropped them, so a parity-verified eval must not grow them, and a
fine-tune that wants them warm-starts from the 5B checkpoint via
:func:`load_choice_head_warm_start`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange, reduce
from torch import Tensor, nn

from opentau.policies.xr1.dit import XR1MLPProjector

#: Reference token ids. Asserted at build time exactly as the reference's collate does --
#: a tokenizer that assigns different ids silently trains three heads on the wrong rows.
SCORE_TOKEN_ID = 151669
STATE_TOKEN_ID = 151670
ACTION_TOKEN_START_ID = STATE_TOKEN_ID + 1
ACTION_TOKEN_END_ID = ACTION_TOKEN_START_ID + 60

IM_START_TOKEN_ID = 151644

SPECIAL_TOKENS = {"score": "<score>", "state": "<state>"}
SPECIAL_TOKENS.update({f"a_{index}": f"<a_{index}>" for index in range(60)})

#: The choice turn, appended after the observation prompt. ``{action_tokens}`` is
#: ``<a_0>...<a_{chunk-1}>``.
CHOICE_TURN_TEMPLATE = (
    "<|im_start|>user\nRobot state: <state><|im_end|>\n"
    "<|im_start|>assistant\n{action_tokens}<score><|im_end|>\n"
)


def render_choice_turn(chunk_size: int) -> str:
    """The choice turn's text for a ``chunk_size``-step action chunk.

    Raises:
        ValueError: if ``chunk_size`` exceeds the 60 available ``<a_i>`` tokens.
    """
    available = ACTION_TOKEN_END_ID - ACTION_TOKEN_START_ID
    if chunk_size > available:
        raise ValueError(
            f"chunk_size ({chunk_size}) exceeds the {available} <a_i> tokens the reference "
            "tokenizer defines; the choice heads use one token per action *timestep*."
        )
    action_tokens = "".join(f"<a_{index}>" for index in range(chunk_size))
    return CHOICE_TURN_TEMPLATE.format(action_tokens=action_tokens)


def assert_reference_token_ids(tokenizer) -> None:
    """Refuse a tokenizer whose special-token ids differ from the reference's.

    Verbatim in spirit from ``custom_collate.CustomCollate.__init__``: the heads index
    hidden states by *id range*, so a shifted vocabulary trains them on unrelated tokens
    without any shape error.

    Raises:
        ValueError: on any mismatch.
    """
    ids = tokenizer.convert_tokens_to_ids(["<score>", "<state>", "<a_0>", "<a_59>"])
    expected = [SCORE_TOKEN_ID, STATE_TOKEN_ID, ACTION_TOKEN_START_ID, ACTION_TOKEN_END_ID - 1]
    if list(ids) != expected:
        raise ValueError(
            f"Choice-token ids {list(ids)} != the reference's {expected}. Add the extra "
            "special tokens when loading the tokenizer "
            "(`extra_special_tokens=SPECIAL_TOKENS`) so the ids line up."
        )


def choice_turn_prefix_length(input_ids: Tensor) -> Tensor:
    """Per-sample length of the observation prefix, i.e. where the choice turn begins.

    The cut is the **last** ``<|im_start|>`` before ``<state>`` -- the reference's rule. The
    DiT's KV cache and cross-attention mask are sliced to this so the choice turn is
    invisible to the action head.

    Args:
        input_ids: ``(B, S)``.

    Returns:
        ``(B,)`` long tensor of prefix lengths.

    Raises:
        ValueError: if any sample has no ``<state>`` token, or no ``<|im_start|>`` before it.
    """
    lengths = []
    for row in input_ids:
        state_positions = (row == STATE_TOKEN_ID).nonzero(as_tuple=False).flatten()
        if state_positions.numel() == 0:
            raise ValueError("A choice-head batch must contain a <state> token per sample.")
        starts = (row == IM_START_TOKEN_ID).nonzero(as_tuple=False).flatten()
        starts = starts[starts < state_positions[0]]
        if starts.numel() == 0:
            raise ValueError("Cannot locate the action-conditioning turn (no <|im_start|>).")
        lengths.append(int(starts[-1]))
    return torch.tensor(lengths, dtype=torch.long, device=input_ids.device)


class XR1ChoiceHeads(nn.Module):
    """The three projectors plus the ``<a_i>`` / ``<score>`` token embeddings.

    Module names mirror the reference's so a warm start from the 5B checkpoint is a plain
    key match (see :func:`load_choice_head_warm_start`). ``action_embed`` / ``score_embed``
    live here rather than on the backbone -- the reference hangs them off its *vendored*
    ``Qwen3VLModel`` as ``vlm.model.action_embed`` / ``vlm.model.score_embed``, and the
    stock class has no such attributes; the warm-start loader carries that rename.
    """

    def __init__(self, *, hidden_size: int, state_dim: int, action_dim: int, n_choices: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.n_choices = n_choices

        self.state_projector_choice = XR1MLPProjector(
            input_dim=state_dim, output_dim=hidden_size, inter_dim=hidden_size, num_layers=2
        )
        self.action_projector_choice = nn.Sequential(
            XR1MLPProjector(hidden_size, hidden_size, inter_dim=hidden_size, num_layers=4),
            XR1MLPProjector(hidden_size, action_dim * n_choices),
        )
        self.score_projector_choice = nn.Sequential(
            XR1MLPProjector(hidden_size, hidden_size, inter_dim=hidden_size, num_layers=4),
            XR1MLPProjector(hidden_size, n_choices),
        )
        self.action_embed = nn.Embedding(ACTION_TOKEN_END_ID - ACTION_TOKEN_START_ID, hidden_size)
        self.score_embed = nn.Embedding(1, hidden_size)

    def build_inputs_embeds(self, input_ids: Tensor, token_embeddings: nn.Module, state: Tensor) -> Tensor:
        """``input_ids`` -> ``inputs_embeds`` with the three choice sources spliced in.

        The ``<|video_pad|>`` rows are deliberately left as the plain token embedding: the
        backbone scatters the vision features into them itself, and with ``input_ids=None``
        it locates them by comparing against ``embed(video_token_id)``, which only works
        because those rows are untouched.

        Args:
            input_ids: ``(B, S)``.
            token_embeddings: The backbone's input-embedding module.
            state: ``(B, T, state_dim)`` adapted state tokens; flattened over ``T`` to fill
                the (single) ``<state>`` row -- the reference projects the whole history and
                takes the last, so ``T`` state rows are pooled by mean here and the pooling
                is stated rather than implicit.

        Returns:
            ``(B, S, hidden)`` embeddings.
        """
        inputs_embeds = token_embeddings(input_ids)

        state_embed = self.state_projector_choice(reduce(state, "b t d -> b d", "mean"))
        state_mask = rearrange(input_ids == STATE_TOKEN_ID, "b s -> b s 1")
        inputs_embeds = inputs_embeds.masked_scatter(
            state_mask, state_embed.to(inputs_embeds.dtype)[state_mask.squeeze(-1).any(dim=-1)]
        )

        action_mask = (input_ids >= ACTION_TOKEN_START_ID) & (input_ids < ACTION_TOKEN_END_ID)
        if bool(action_mask.any()):
            action_embeds = self.action_embed(input_ids[action_mask] - ACTION_TOKEN_START_ID)
            inputs_embeds = inputs_embeds.masked_scatter(
                rearrange(action_mask, "b s -> b s 1"), action_embeds.to(inputs_embeds.dtype)
            )

        score_mask = input_ids == SCORE_TOKEN_ID
        if bool(score_mask.any()):
            score_embeds = self.score_embed(torch.zeros_like(input_ids[score_mask]))
            inputs_embeds = inputs_embeds.masked_scatter(
                rearrange(score_mask, "b s -> b s 1"), score_embeds.to(inputs_embeds.dtype)
            )
        return inputs_embeds

    def predict(self, hidden_states: Tensor, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Read the ``<a_i>`` and ``<score>`` rows and project them.

        Returns:
            ``(action_pred, score_pred)`` of shapes ``(B, chunk, n_choices, action_dim)``
            and ``(B, n_choices)``.
        """
        batch_size = input_ids.shape[0]
        action_mask = (input_ids >= ACTION_TOKEN_START_ID) & (input_ids < ACTION_TOKEN_END_ID)
        counts = action_mask.sum(dim=1)
        if not bool((counts == counts[0]).all()):
            raise ValueError(
                f"Every sample must carry the same number of <a_i> tokens (got {counts.tolist()}); "
                "the choice heads are vectorized over a fixed chunk length."
            )
        chunk = int(counts[0])
        action_hidden = rearrange(hidden_states[action_mask], "(b c) h -> b c h", b=batch_size, c=chunk)
        action_pred = self.action_projector_choice(action_hidden)
        action_pred = rearrange(action_pred, "b c (n d) -> b c n d", n=self.n_choices, d=self.action_dim)

        score_mask = input_ids == SCORE_TOKEN_ID
        score_hidden = rearrange(hidden_states[score_mask], "(b one) h -> b one h", b=batch_size)[:, 0]
        score_pred = self.score_projector_choice(score_hidden)
        return action_pred, score_pred


def choice_loss(
    action_pred: Tensor, score_pred: Tensor, target: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Winner-takes-all L1 on the candidates, plus MSE on the predicted errors.

    Vectorized over the batch. The reference loops per sample because its collate *packs*
    variable-length samples into one sequence; with OpenTau's padded batching every sample
    has the same chunk length, so the loop is unnecessary -- and
    ``test_xr1_cpu.py`` pins the vectorized form against a literal transcription of the
    loop.

    Args:
        action_pred: ``(B, chunk, n_choices, action_dim)``.
        score_pred: ``(B, n_choices)``.
        target: ``(B, chunk, action_dim)`` ground-truth actions.
        mask: ``(B, chunk, action_dim)`` bool; True where the target is real.

    Returns:
        ``(loss_choice, loss_score)``, both scalars.

    Raises:
        ValueError: if any sample has no valid target value (the reference raises too --
            a sample with nothing to supervise means the batch is malformed, not empty).
    """
    action_pred = action_pred.float()
    target = rearrange(target.float(), "b c d -> b c 1 d")
    mask = rearrange(mask.bool(), "b c d -> b c 1 d")
    if not bool(mask.any(dim=(1, 3)).all()):
        raise ValueError("A choice-loss sample has no valid action values to supervise.")

    absolute_error = (action_pred - target).abs() * mask
    # Per-candidate mean over the valid slots of that sample.
    denominator = reduce(mask.float(), "b c one d -> b one", "sum").clamp_min(1.0)
    per_choice = reduce(absolute_error, "b c n d -> b n", "sum") / denominator
    best = per_choice.min(dim=1)
    loss_choice = best.values.mean()
    loss_score = F.mse_loss(score_pred.float(), per_choice.detach())
    return loss_choice, loss_score


#: ``reference key -> ours``, for the 5B checkpoint's choice-head tensors. The three
#: projectors keep their names; only the two token embeddings move, because the reference
#: hangs them off its vendored ``Qwen3VLModel`` and ours is stock.
WARM_START_RENAMES = {
    "vlm.model.action_embed.weight": "action_embed.weight",
    "vlm.model.score_embed.weight": "score_embed.weight",
}


def load_choice_head_warm_start(heads: XR1ChoiceHeads, state_dict: dict[str, Tensor]) -> list[str]:
    """Copy the choice-head tensors out of a Xiaomi-Robotics-1-5B state dict.

    Only the 15 tensors these heads own are read; everything else in the source is ignored,
    so this can be pointed straight at ``ckpt_5b/model_states.pt``.

    Returns:
        The keys that were loaded, so a caller can log (and a test assert) the count.

    Raises:
        ValueError: if a matched tensor has the wrong shape -- a silent skip there would
            leave a randomly-initialized head that looks warm-started.
    """
    own = heads.state_dict()
    loaded: dict[str, Tensor] = {}
    for key, tensor in state_dict.items():
        target = WARM_START_RENAMES.get(key, key)
        if target not in own:
            continue
        if tuple(tensor.shape) != tuple(own[target].shape):
            raise ValueError(
                f"Warm-start shape mismatch for '{key}' -> '{target}': {tuple(tensor.shape)} vs "
                f"{tuple(own[target].shape)}. Refusing to skip it silently."
            )
        loaded[target] = tensor
    heads.load_state_dict(loaded, strict=False)
    return sorted(loaded)
