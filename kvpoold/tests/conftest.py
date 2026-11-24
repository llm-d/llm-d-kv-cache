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

import pytest
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import EventPublisherFactory

from .recording_publisher import PUBLISHER_NAME, TOPIC, RecordingPublisher


@pytest.fixture
def recording_config():
    """A ``KVEventsConfig`` that names ``RecordingPublisher``, registered with vLLM's
    factory for the test. A tier built with it needs no socket for its events."""
    config = KVEventsConfig(enable_kv_cache_events=True, topic=TOPIC)
    config.publisher = PUBLISHER_NAME  # the field's type admits only vLLM's own publishers, at construction
    EventPublisherFactory.register_publisher(PUBLISHER_NAME, RecordingPublisher)
    try:
        yield config
    finally:
        del EventPublisherFactory._registry[PUBLISHER_NAME]  # the factory has no unregister
