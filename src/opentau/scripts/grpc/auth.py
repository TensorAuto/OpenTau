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

"""Optional API-key authentication for the robot inference gRPC server.

Authentication is opt-in: it is activated only when the
``TUNER_INFERENCE_API_KEY`` environment variable is set to a non-empty value.
When active, every RPC must carry the matching key in the ``x-tuner-api-key``
metadata header; otherwise the call is rejected with ``UNAUTHENTICATED``.
When the variable is unset/empty the server runs with no authentication (the
historical behavior), so this is fully backward compatible.

This module deliberately depends only on ``grpc`` (not torch / the policy
stack) so the auth logic can be unit-tested cheaply, CPU-only.
"""

from __future__ import annotations

import logging
import os

import grpc

logger = logging.getLogger(__name__)

API_KEY_HEADER = "x-tuner-api-key"
API_KEY_ENV = "TUNER_INFERENCE_API_KEY"


def extract_api_key(metadata) -> str | None:
    """Return the ``x-tuner-api-key`` value from gRPC invocation metadata.

    Args:
        metadata: Iterable of ``(key, value)`` pairs (gRPC invocation
            metadata), or ``None``.

    Returns:
        The api key string if present, else ``None``.
    """
    if not metadata:
        return None
    for key, value in metadata:
        if key == API_KEY_HEADER:
            return value
    return None


def is_authorized(metadata, expected_key: str) -> bool:
    """Whether ``metadata`` carries the expected api key."""
    provided = extract_api_key(metadata)
    return provided is not None and provided == expected_key


class ApiKeyInterceptor(grpc.ServerInterceptor):
    """Server interceptor that requires a valid ``x-tuner-api-key`` header.

    RPCs whose metadata is missing the header or carries a wrong key are
    aborted with ``grpc.StatusCode.UNAUTHENTICATED`` before reaching the
    servicer.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key

        def _abort(request, context):
            context.abort(
                grpc.StatusCode.UNAUTHENTICATED,
                f"Missing or invalid {API_KEY_HEADER}",
            )

        self._deny = grpc.unary_unary_rpc_method_handler(_abort)

    def intercept_service(self, continuation, handler_call_details):
        if is_authorized(handler_call_details.invocation_metadata, self._api_key):
            return continuation(handler_call_details)
        return self._deny


def interceptor_from_env() -> ApiKeyInterceptor | None:
    """Build an :class:`ApiKeyInterceptor` from ``TUNER_INFERENCE_API_KEY``.

    Returns:
        An interceptor when the env var is set to a non-empty (stripped)
        value, else ``None`` (authentication disabled).
    """
    api_key = os.getenv(API_KEY_ENV, "").strip()
    if not api_key:
        return None
    return ApiKeyInterceptor(api_key)
