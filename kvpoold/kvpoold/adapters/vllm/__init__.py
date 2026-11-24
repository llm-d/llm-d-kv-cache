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

"""The vLLM adapter: an offloading spec and manager over the pool server's client.

vLLM gives a handler only to its own ``vllm`` logger, which does not propagate,
and it may set that handler up after plugins load. Importing this package gives
the ``kvpoold`` logger one handler that passes each record to the ``vllm``
logger when it is emitted, so the lines of every kvpoold module in an engine
process reach the engine's log with vLLM's format and level, whenever vLLM
configures it. Both the ``vllm.general_plugins`` entry point and a
``spec_module_path`` import this package before any of its modules logs.
"""

import logging


class _ToVllmLog(logging.Handler):
    """Passes a record to the ``vllm`` logger's handlers if vLLM's level lets it through."""

    def emit(self, record: logging.LogRecord) -> None:
        vllm_logger = logging.getLogger("vllm")
        if record.levelno >= vllm_logger.getEffectiveLevel():
            vllm_logger.handle(record)


def _route_logs_to_vllm() -> None:
    kvpoold_logger = logging.getLogger("kvpoold")
    if any(isinstance(handler, _ToVllmLog) for handler in kvpoold_logger.handlers):
        return
    kvpoold_logger.addHandler(_ToVllmLog())
    # vLLM's level decides what is logged, so this logger lets every record through.
    kvpoold_logger.setLevel(logging.DEBUG)
    kvpoold_logger.propagate = False


_route_logs_to_vllm()
