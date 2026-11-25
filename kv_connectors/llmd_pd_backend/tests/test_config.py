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

"""Unit tests for P/D tier config validation."""

import pytest

from llmd_pd_backend.config import (
    DEFAULT_CONNECT_TIMEOUT_S,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_BACKOFF_S,
    DEFAULT_TRANSFER_TIMEOUT_S,
    validate_pd_config,
)


class TestValidateConfig:
    def test_valid_config(self):
        cfg = validate_pd_config({"host": "10.0.0.1", "port": 5710})
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 5710
        assert cfg.connect_timeout_s == DEFAULT_CONNECT_TIMEOUT_S
        assert cfg.transfer_timeout_s == DEFAULT_TRANSFER_TIMEOUT_S
        assert cfg.max_retries == DEFAULT_MAX_RETRIES
        assert cfg.retry_backoff_s == DEFAULT_RETRY_BACKOFF_S

    def test_valid_config_with_overrides(self):
        cfg = validate_pd_config(
            {
                "host": "10.0.0.1",
                "port": 5710,
                "connect_timeout_s": 5.0,
                "transfer_timeout_s": 60.0,
                "max_retries": 5,
                "retry_backoff_s": 2.0,
            }
        )
        assert cfg.connect_timeout_s == 5.0
        assert cfg.transfer_timeout_s == 60.0
        assert cfg.max_retries == 5
        assert cfg.retry_backoff_s == 2.0

    def test_missing_host_raises(self):
        with pytest.raises(ValueError, match="requires 'host'"):
            validate_pd_config({"port": 5710})

    def test_empty_host_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_pd_config({"host": "", "port": 5710})

    def test_missing_port_raises(self):
        with pytest.raises(ValueError, match="requires 'port'"):
            validate_pd_config({"host": "10.0.0.1"})

    def test_invalid_port_type_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            validate_pd_config({"host": "10.0.0.1", "port": "5710"})

    def test_bool_port_raises(self):
        with pytest.raises(ValueError, match="must be an integer"):
            validate_pd_config({"host": "10.0.0.1", "port": True})

    def test_port_out_of_range_raises(self):
        with pytest.raises(ValueError, match="1-65535"):
            validate_pd_config({"host": "10.0.0.1", "port": 0})
        with pytest.raises(ValueError, match="1-65535"):
            validate_pd_config({"host": "10.0.0.1", "port": 70000})

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="positive"):
            validate_pd_config(
                {
                    "host": "10.0.0.1",
                    "port": 5710,
                    "connect_timeout_s": -1.0,
                }
            )

    def test_negative_transfer_timeout_raises(self):
        with pytest.raises(ValueError, match="positive"):
            validate_pd_config(
                {
                    "host": "10.0.0.1",
                    "port": 5710,
                    "transfer_timeout_s": -1.0,
                }
            )

    def test_zero_max_retries_raises(self):
        with pytest.raises(ValueError, match="positive"):
            validate_pd_config(
                {
                    "host": "10.0.0.1",
                    "port": 5710,
                    "max_retries": 0,
                }
            )

    def test_negative_backoff_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_pd_config(
                {
                    "host": "10.0.0.1",
                    "port": 5710,
                    "retry_backoff_s": -0.5,
                }
            )

    def test_defaults_applied(self):
        cfg = validate_pd_config({"host": "localhost", "port": 8080})
        assert cfg.connect_timeout_s == DEFAULT_CONNECT_TIMEOUT_S
        assert cfg.transfer_timeout_s == DEFAULT_TRANSFER_TIMEOUT_S
        assert cfg.max_retries == DEFAULT_MAX_RETRIES
        assert cfg.retry_backoff_s == DEFAULT_RETRY_BACKOFF_S

    def test_config_is_frozen(self):
        cfg = validate_pd_config({"host": "10.0.0.1", "port": 5710})
        with pytest.raises(AttributeError):
            cfg.host = "changed"
