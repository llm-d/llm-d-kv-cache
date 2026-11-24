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

"""The vLLM adapter routes kvpoold's logs into vLLM's log."""

import logging

import pytest

import kvpoold.adapters.vllm  # noqa: F401  (routes the logs on import)


class Recording(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append((record.name, record.getMessage()))


@pytest.fixture
def vllm_handler():
    """The ``vllm`` logger as vLLM's ``configure_logging`` leaves it, after the
    adapter was imported: one handler, at INFO, not propagating."""
    vllm_logger = logging.getLogger("vllm")
    handler, level, propagate = Recording(), vllm_logger.level, vllm_logger.propagate
    vllm_logger.addHandler(handler)
    vllm_logger.setLevel(logging.INFO)
    vllm_logger.propagate = False
    try:
        yield handler
    finally:
        vllm_logger.removeHandler(handler)
        vllm_logger.setLevel(level)
        vllm_logger.propagate = propagate


def test_a_kvpoold_line_reaches_vllms_handlers(vllm_handler):
    logging.getLogger("kvpoold.client.client").info("attached again")
    assert vllm_handler.lines == [("kvpoold.client.client", "attached again")]


def test_vllms_level_decides_what_is_logged(vllm_handler):
    logging.getLogger("kvpoold.client.session").debug("dropped a notice")
    assert vllm_handler.lines == []


def test_a_kvpoold_line_does_not_also_reach_the_root_loggers_handlers(vllm_handler):
    root, root_handler = logging.getLogger(), Recording()
    root.addHandler(root_handler)
    try:
        logging.getLogger("kvpoold.client.client").warning("serving without the pool")
    finally:
        root.removeHandler(root_handler)
    assert root_handler.lines == []
    assert vllm_handler.lines == [("kvpoold.client.client", "serving without the pool")]
