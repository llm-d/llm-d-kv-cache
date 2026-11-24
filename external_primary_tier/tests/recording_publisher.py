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

"""A KV events publisher for tests that need a tier but do not look at its events."""

from vllm.distributed.kv_events import EventPublisher, KVEventBatch

TOPIC = "kv@node-1@org/model-a"
PUBLISHER_NAME = "recording"


class RecordingPublisher(EventPublisher):
    """Records what the tier does with its publisher, in order. Built the way vLLM's
    factory builds a registered publisher: from the config's fields as keywords."""

    def __init__(self, **config_fields) -> None:
        super().__init__(config_fields.pop("data_parallel_rank", 0))
        self.config_fields = config_fields
        self.calls: list[tuple[str, list]] = []

    def publish(self, events: KVEventBatch) -> None:
        self.calls.append(("publish", events.events))

    def shutdown(self) -> None:
        self.calls.append(("shutdown", []))
