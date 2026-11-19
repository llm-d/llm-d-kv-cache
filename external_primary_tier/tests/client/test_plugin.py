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

"""Registration under vLLM's ``vllm.general_plugins`` entry point."""

from external_primary_tier.client import plugin
from external_primary_tier.client.spec import ExternalOffloadingSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory


def test_spec_name_alone_selects_the_spec():
    plugin.register()
    assert OffloadingSpecFactory.get_spec_cls({"spec_name": plugin.SPEC_NAME}) is ExternalOffloadingSpec


def test_registering_again_is_harmless():
    plugin.register()
    plugin.register()


def test_register_never_raises(monkeypatch):
    """vLLM calls entry points unguarded, so a failure would stop every process from starting."""

    def fail(*args, **kwargs):
        raise RuntimeError("the registry changed")

    monkeypatch.setattr(OffloadingSpecFactory, "register_spec", fail)
    plugin.register()
