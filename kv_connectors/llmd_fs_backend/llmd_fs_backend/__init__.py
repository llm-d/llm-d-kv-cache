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

import logging
import os

from vllm.logger import init_logger

from llmd_fs_backend.metrics import install_offload_metric_suffix_patch

# Apply the OffloadPromMetrics spec-suffix patch before any OffloadingConnector
# is instantiated. Required so MultiConnector can nest two OffloadingConnector
# backends without hitting "Duplicated timeseries in CollectorRegistry".
# TODO: remove once vLLM upstream applies an equivalent spec_name suffix
# to the OffloadPromMetrics names.
install_offload_metric_suffix_patch()

_LEVEL_MAP = {
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

# STORAGE_LOG_LEVEL controls log level (default: INFO).
_log_level_str = os.environ.get("STORAGE_LOG_LEVEL", "INFO").upper()
_logger = init_logger(__name__)
_logger.setLevel(_LEVEL_MAP.get(_log_level_str, logging.INFO))

# Ensure logger has a handler. vllm's init_logger creates loggers without handlers
if not _logger.handlers:
    _handler = logging.StreamHandler()
    # Set logger format
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _logger.addHandler(_handler)
