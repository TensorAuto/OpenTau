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

"""High-level planner backed by Gemini Robotics-ER.

The planner implements the "memory-as-language" pattern: each call receives the
overall task, the memory string the model wrote on the previous call, the
current camera images and (optionally) the robot proprioceptive state, and
returns a rewritten memory plus the next subtask. The subtask is phrased as a
short imperative language command suitable for conditioning a low-level VLA
policy.

Example:
    >>> from opentau.planner import GeminiERPlanner
    >>> planner = GeminiERPlanner()  # needs GEMINI_API_KEY in the environment
    >>> result = planner.plan(
    ...     task="put all the blocks in the bin",
    ...     images=[jpeg_bytes],
    ...     state=[0.0] * 7,
    ...     memory="",
    ... )
    >>> result.subtask, result.memory
"""

import io
import json
import logging
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange
from google import genai
from google.genai import types as genai_types
from PIL import Image

from opentau.planner.utils.utils import load_prompt_library

logger = logging.getLogger(__name__)

# Image types accepted by `GeminiERPlanner.plan`: encoded bytes (assumed JPEG),
# a `(bytes, encoding)` tuple ("jpeg" or "png"), a PIL image, or a float tensor
# of shape (3, H, W) or (1, 3, H, W) with values in [0, 1].
PlannerImage = bytes | tuple[bytes, str] | Image.Image | torch.Tensor

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

_PLAN_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "memory": {"type": "STRING"},
        "subtask": {"type": "STRING"},
    },
    "required": ["memory", "subtask"],
}


@dataclass(frozen=True)
class PlanResult:
    """Result of one high-level planning call.

    Args:
        subtask: The next subtask for the low-level VLA policy.
        memory: The rewritten memory-as-language string.
        raw_text: Raw model output text, for logging/debugging.
        latency_s: Wall-clock seconds the model call took.
    """

    subtask: str
    memory: str
    raw_text: str
    latency_s: float


def _parse_plan_response(text: str, prev_memory: str) -> tuple[str, str]:
    """Parse the planner's JSON response into ``(subtask, memory)``.

    Args:
        text: Raw model output, possibly wrapped in markdown code fences.
        prev_memory: Memory from the previous call, used as a fallback when the
            response omits the ``memory`` key.

    Returns:
        Tuple of ``(subtask, memory)``.

    Raises:
        ValueError: If the response is not valid JSON or lacks a ``subtask``.
    """
    cleaned = _JSON_FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner response is not valid JSON: {text!r}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"Planner response is not a JSON object: {text!r}")

    subtask = parsed.get("subtask")
    if not isinstance(subtask, str) or not subtask.strip():
        raise ValueError(f"Planner response has no 'subtask': {text!r}")

    memory = parsed.get("memory")
    if not isinstance(memory, str) or not memory.strip():
        memory = prev_memory

    return subtask.strip(), memory.strip()


def _to_part(image: PlannerImage) -> genai_types.Part:
    """Convert a supported image input into a google-genai content ``Part``.

    Encoded bytes are forwarded verbatim (no re-encode); PIL images and float
    tensors are JPEG-encoded.
    """
    if isinstance(image, bytes):
        return genai_types.Part.from_bytes(data=image, mime_type="image/jpeg")
    if isinstance(image, tuple):
        data, encoding = image
        return genai_types.Part.from_bytes(data=data, mime_type=f"image/{encoding.lower()}")
    if isinstance(image, torch.Tensor):
        tensor = rearrange(image.detach(), "1 c h w -> c h w") if image.dim() == 4 else image.detach()
        if tensor.dim() != 3:
            raise ValueError(f"Expected image tensor of shape (3, H, W) or (1, 3, H, W), got {image.shape}")
        tensor = (tensor.to(dtype=torch.float32, device="cpu").clamp(0, 1) * 255.0).to(torch.uint8)
        image = Image.fromarray(rearrange(tensor, "c h w -> h w c").numpy())
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        return genai_types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/jpeg")
    raise TypeError(f"Unsupported planner image type: {type(image)}")


class GeminiERPlanner:
    """High-level planner that queries Gemini Robotics-ER for the next subtask.

    Unlike the conversation-history planners in ``high_level_planner.py``, the
    only persistent state is the memory string the model itself rewrites on
    every call — the caller owns it and passes it back in.
    """

    DEFAULT_MODEL = "gemini-robotics-er-1.5-preview"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        api_key_env: str = "GEMINI_API_KEY",
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt_key: str = "gemini_er_planner_system",
        user_prompt_key: str = "gemini_er_planner_user",
        client: genai.Client | None = None,
    ):
        """Initialize the planner.

        Args:
            model: Gemini model ID.
            api_key: Explicit API key. When None, ``api_key_env`` then
                ``GOOGLE_API_KEY`` are tried.
            api_key_env: Environment variable to read the API key from.
            max_output_tokens: Generation cap for the planner response.
            temperature: Sampling temperature.
            system_prompt_key: Key into ``planner/prompts.yaml`` for the system prompt.
            user_prompt_key: Key into ``planner/prompts.yaml`` for the user prompt.
            client: Pre-built genai client (used in tests). When provided,
                no API key is required.

        Raises:
            ValueError: If no API key is resolvable or a prompt key is missing.
        """
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

        prompts_path = Path(__file__).resolve().parent / "prompts.yaml"
        prompt_library = load_prompt_library(str(prompts_path))
        if not prompt_library:
            raise ValueError(f"Could not load prompt library from {prompts_path}")
        prompts = prompt_library["prompts"]
        for key in (system_prompt_key, user_prompt_key):
            if key not in prompts:
                raise ValueError(f"Prompt key '{key}' not found in {prompts_path}")
        self.system_prompt = prompts[system_prompt_key]["template"]
        self.user_prompt_template = prompts[user_prompt_key]["template"]

        if client is not None:
            self.client = client
        else:
            api_key = api_key or os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    f"No Gemini API key found: pass `api_key` or set ${api_key_env} (or $GOOGLE_API_KEY)."
                )
            self.client = genai.Client(api_key=api_key)

        # Structured-output support is unverified on the ER preview family;
        # fall back to mime-type-only JSON once if the API rejects the schema.
        self._schema_supported = True

    def _build_user_prompt(self, task: str, memory: str, state: Sequence[float] | None) -> str:
        """Fill the user prompt template with task, memory and state."""
        state_str = "(not provided)" if state is None else str([round(float(s), 4) for s in state])
        return self.user_prompt_template.format(
            task=task,
            memory=memory.strip() or "(empty — first call)",
            state=state_str,
        )

    def _generate(self, contents: list) -> str:
        """Call the model, retrying once without ``response_schema`` if rejected."""
        config_kwargs = {
            "system_instruction": self.system_prompt,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": "application/json",
        }
        if self._schema_supported:
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        **config_kwargs, response_schema=_PLAN_RESPONSE_SCHEMA
                    ),
                )
                return self._response_text(response)
            except genai.errors.APIError as e:
                if "schema" not in str(e).lower():
                    raise
                logger.warning("Model %s rejected response_schema; retrying without it: %s", self.model, e)
                self._schema_supported = False

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(**config_kwargs),
        )
        return self._response_text(response)

    @staticmethod
    def _response_text(response) -> str:
        text = (response.text or "").strip()
        if not text:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            raise ValueError(f"Planner response had no text (finish_reason={finish_reason}).")
        return text

    def plan(
        self,
        task: str,
        images: Sequence[PlannerImage],
        state: Sequence[float] | None = None,
        memory: str = "",
    ) -> PlanResult:
        """Run one planning step.

        Args:
            task: The overall task the robot must complete.
            images: Current camera images (see ``PlannerImage`` for accepted types).
            state: Optional robot proprioceptive state vector.
            memory: Memory string returned by the previous call ("" on the first).

        Returns:
            PlanResult with the next subtask and the rewritten memory.

        Raises:
            ValueError: If the model response cannot be parsed into a plan.
        """
        contents: list = [self._build_user_prompt(task, memory, state)]
        contents.extend(_to_part(image) for image in images)

        start = time.perf_counter()
        raw_text = self._generate(contents)
        latency_s = time.perf_counter() - start

        subtask, new_memory = _parse_plan_response(raw_text, memory)
        return PlanResult(subtask=subtask, memory=new_memory, raw_text=raw_text, latency_s=latency_s)
