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

"""Tests for the optional API-key interceptor (``auth.py``).

CPU-only, no torch / policy stack: the interceptor is exercised against a
throwaway in-process gRPC server with a generic echo handler.
"""

from concurrent import futures

import grpc
import pytest

from opentau.scripts.grpc import auth


def test_extract_api_key():
    md = (("other", "x"), (auth.API_KEY_HEADER, "secret"))
    assert auth.extract_api_key(md) == "secret"
    assert auth.extract_api_key(()) is None
    assert auth.extract_api_key(None) is None


def test_is_authorized():
    md = ((auth.API_KEY_HEADER, "secret"),)
    assert auth.is_authorized(md, "secret") is True
    assert auth.is_authorized(md, "wrong") is False
    assert auth.is_authorized((), "secret") is False


def test_interceptor_from_env(monkeypatch):
    monkeypatch.delenv(auth.API_KEY_ENV, raising=False)
    assert auth.interceptor_from_env() is None

    monkeypatch.setenv(auth.API_KEY_ENV, "   ")  # whitespace == unset
    assert auth.interceptor_from_env() is None

    monkeypatch.setenv(auth.API_KEY_ENV, "k")
    assert isinstance(auth.interceptor_from_env(), auth.ApiKeyInterceptor)


def test_interceptor_requires_non_empty_key():
    with pytest.raises(ValueError):
        auth.ApiKeyInterceptor("")


def _start_echo_server(interceptor):
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=1),
        interceptors=[interceptor],
    )

    def _echo(request, context):
        return request

    def _echo_stream(request_iterator, context):
        yield from request_iterator

    handlers = {
        # unary_unary RPC.
        "Ping": grpc.unary_unary_rpc_method_handler(
            _echo,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        ),
        # stream_stream RPC, mirroring the real server's StreamActionChunks.
        "PingStream": grpc.stream_stream_rpc_method_handler(
            _echo_stream,
            request_deserializer=lambda b: b,
            response_serializer=lambda b: b,
        ),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("test.Echo", handlers),)
    )
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port


def _ping(port, metadata):
    with grpc.insecure_channel(f"localhost:{port}") as channel:
        call = channel.unary_unary("/test.Echo/Ping")
        return call(b"hello", metadata=metadata)


def _ping_stream(port, metadata):
    with grpc.insecure_channel(f"localhost:{port}") as channel:
        call = channel.stream_stream("/test.Echo/PingStream")
        return list(call(iter([b"hello", b"world"]), metadata=metadata))


def test_valid_key_passes_through():
    server, port = _start_echo_server(auth.ApiKeyInterceptor("secret"))
    try:
        assert _ping(port, ((auth.API_KEY_HEADER, "secret"),)) == b"hello"
    finally:
        server.stop(None)


def test_valid_key_passes_through_streaming():
    server, port = _start_echo_server(auth.ApiKeyInterceptor("secret"))
    try:
        assert _ping_stream(port, ((auth.API_KEY_HEADER, "secret"),)) == [b"hello", b"world"]
    finally:
        server.stop(None)


@pytest.mark.parametrize("metadata", [(), (("x-tuner-api-key", "wrong"),)])
def test_missing_or_wrong_key_unauthenticated(metadata):
    server, port = _start_echo_server(auth.ApiKeyInterceptor("secret"))
    try:
        with pytest.raises(grpc.RpcError) as exc:
            _ping(port, metadata)
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        server.stop(None)


@pytest.mark.parametrize("metadata", [(), (("x-tuner-api-key", "wrong"),)])
def test_missing_or_wrong_key_unauthenticated_streaming(metadata):
    # The deny handler is unary_unary only, but context.abort() fires before the
    # request stream is read, so it must reject stream_stream RPCs too.
    server, port = _start_echo_server(auth.ApiKeyInterceptor("secret"))
    try:
        with pytest.raises(grpc.RpcError) as exc:
            _ping_stream(port, metadata)
        assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        server.stop(None)
