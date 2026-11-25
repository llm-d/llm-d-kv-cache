# Copyright 2025 The llm-d Authors.
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

"""Configuration validation for the P/D NIXL secondary tier."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_TRANSFER_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 1.0


@dataclass(frozen=True, slots=True)
class PDConfig:
    host: str
    port: int
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    transfer_timeout_s: float = DEFAULT_TRANSFER_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES
    retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S


def validate_pd_config(raw: dict) -> PDConfig:
    """Validate and return a typed config from the secondary_tiers entry.

    Raises ValueError on invalid or missing required fields.
    """
    if "host" not in raw:
        raise ValueError("P/D tier config requires 'host'")
    host = raw["host"]
    if not isinstance(host, str) or not host:
        raise ValueError(f"'host' must be a non-empty string, got {host!r}")

    if "port" not in raw:
        raise ValueError("P/D tier config requires 'port'")
    port = raw["port"]
    if not isinstance(port, int) or isinstance(port, bool):
        raise ValueError(f"'port' must be an integer, got {type(port).__name__}")
    if port <= 0 or port > 65535:
        raise ValueError(f"'port' must be 1-65535, got {port}")

    connect_timeout_s = float(raw.get("connect_timeout_s", DEFAULT_CONNECT_TIMEOUT_S))
    if connect_timeout_s <= 0:
        raise ValueError(
            f"'connect_timeout_s' must be positive, got {connect_timeout_s}"
        )

    transfer_timeout_s = float(
        raw.get("transfer_timeout_s", DEFAULT_TRANSFER_TIMEOUT_S)
    )
    if transfer_timeout_s <= 0:
        raise ValueError(
            f"'transfer_timeout_s' must be positive, got {transfer_timeout_s}"
        )

    max_retries = int(raw.get("max_retries", DEFAULT_MAX_RETRIES))
    if max_retries <= 0:
        raise ValueError(f"'max_retries' must be positive, got {max_retries}")

    retry_backoff_s = float(raw.get("retry_backoff_s", DEFAULT_RETRY_BACKOFF_S))
    if retry_backoff_s < 0:
        raise ValueError(
            f"'retry_backoff_s' must be non-negative, got {retry_backoff_s}"
        )

    return PDConfig(
        host=host,
        port=port,
        connect_timeout_s=connect_timeout_s,
        transfer_timeout_s=transfer_timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
