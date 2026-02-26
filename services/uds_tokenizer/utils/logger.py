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
from collections.abc import Hashable
from functools import lru_cache
from logging import Logger
from types import MethodType
from typing import cast


@lru_cache
def _print_info_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print info message only once"""
    logger.info(msg, *args, stacklevel=2)


@lru_cache
def _print_warning_once(logger: Logger, msg: str, *args: Hashable) -> None:
    """Print warning message only once"""
    logger.warning(msg, *args, stacklevel=2)


class _TokenizerLogger(Logger):
    """
    Extended logger with additional methods for one-time messages.
    """

    def info_once(self, msg: str, *args: Hashable) -> None:
        """
        Log info message only once.
        Subsequent calls with the same message are silently dropped.
        """
        _print_info_once(self, msg, *args)

    def warning_once(self, msg: str, *args: Hashable) -> None:
        """
        Log warning message only once.
        Subsequent calls with the same message are silently dropped.
        """
        _print_warning_once(self, msg, *args)


def init_logger(name: str) -> _TokenizerLogger:
    """Initialize logger with extended functionality"""
    logger = logging.getLogger(name)

    # Add custom methods
    methods_to_patch = {
        "info_once": _print_info_once,
        "warning_once": _print_warning_once,
    }

    for method_name, method in methods_to_patch.items():
        setattr(logger, method_name, MethodType(method, logger))

    return cast(_TokenizerLogger, logger)
