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
"""Deployment configuration classes for inference servers.

This module provides configuration classes for deploying trained models
as inference servers, including gRPC server settings.
"""

from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for the gRPC inference server.

    This class contains all configuration parameters needed to run a gRPC
    inference server for robot policy models.

    Args:
        port: Port number to serve on. Must be between 1 and 65535.
            Defaults to 50051.
        max_workers: Maximum number of gRPC worker threads for handling
            concurrent requests. Defaults to 4.
        max_send_message_length_mb: Maximum size of outgoing messages in
            megabytes. Defaults to 100.
        max_receive_message_length_mb: Maximum size of incoming messages in
            megabytes. Defaults to 100.

    Raises:
        ValueError: If port is not in valid range or max_workers is less than 1.

    Example:
        >>> config = ServerConfig(port=50051, max_workers=8)
        >>> config.port
        50051
    """

    port: int = 50051
    max_workers: int = 4
    max_send_message_length_mb: int = 100
    max_receive_message_length_mb: int = 100

    def __post_init__(self):
        """Validate server configuration parameters."""
        if not 1 <= self.port <= 65535:
            raise ValueError(f"`port` must be between 1 and 65535, got {self.port}.")
        if self.max_workers < 1:
            raise ValueError(f"`max_workers` must be at least 1, got {self.max_workers}.")
        if self.max_send_message_length_mb < 1:
            raise ValueError(
                f"`max_send_message_length_mb` must be at least 1, got {self.max_send_message_length_mb}."
            )
        if self.max_receive_message_length_mb < 1:
            raise ValueError(
                f"`max_receive_message_length_mb` must be at least 1, got {self.max_receive_message_length_mb}."
            )

    @property
    def max_send_message_length(self) -> int:
        """Get maximum send message length in bytes.

        Returns:
            Maximum send message length in bytes.
        """
        return self.max_send_message_length_mb * 1024 * 1024

    @property
    def max_receive_message_length(self) -> int:
        """Get maximum receive message length in bytes.

        Returns:
            Maximum receive message length in bytes.
        """
        return self.max_receive_message_length_mb * 1024 * 1024
