# Copyright 2026 The llm-d Authors.
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

"""Unit tests for the extended logger's init and one-time-message helpers."""

import logging

from utils.logger import init_logger


def test_init_logger_returns_logger_with_given_name():
    logger = init_logger("test.logger.name")

    assert logger.name == "test.logger.name"
    assert isinstance(logger, logging.Logger)


def test_init_logger_attaches_once_methods():
    logger = init_logger("test.logger.once.methods")

    assert hasattr(logger, "info_once")
    assert hasattr(logger, "warning_once")
    assert callable(logger.info_once)
    assert callable(logger.warning_once)


def test_info_once_logs_only_first_call(caplog):
    logger = init_logger("test.logger.info.once")

    with caplog.at_level(logging.INFO, logger="test.logger.info.once"):
        logger.info_once("hello %s", "world")
        logger.info_once("hello %s", "world")

    matching = [r for r in caplog.records if r.message == "hello world"]
    assert len(matching) == 1


def test_warning_once_logs_only_first_call(caplog):
    logger = init_logger("test.logger.warning.once")

    with caplog.at_level(logging.WARNING, logger="test.logger.warning.once"):
        logger.warning_once("careful %s", "now")
        logger.warning_once("careful %s", "now")

    matching = [r for r in caplog.records if r.message == "careful now"]
    assert len(matching) == 1


def test_info_once_distinct_messages_both_log(caplog):
    logger = init_logger("test.logger.info.distinct")

    with caplog.at_level(logging.INFO, logger="test.logger.info.distinct"):
        logger.info_once("first message")
        logger.info_once("second message")

    messages = {r.message for r in caplog.records}
    assert messages == {"first message", "second message"}
