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

"""Register the pool's offloading spec with vLLM at start-up.

vLLM loads this module through the ``vllm.general_plugins`` entry point, in the
scheduler process and in every GPU worker, so ``spec_name`` alone selects the
spec in ``--kv-transfer-config``. Without the entry point, an engine names it
with ``spec_module_path`` instead.
"""

from vllm.logger import init_logger
from vllm.v1.kv_offload.factory import OffloadingSpecFactory

logger = init_logger(__name__)

SPEC_NAME = "ExternalOffloadingSpec"


def register() -> None:
    """Make ``SPEC_NAME`` resolvable.

    Safe to call more than once, and never raises: vLLM calls this unguarded
    while loading plugins, so an exception here would fail the start-up of every
    process, including the ones that never use the pool. A failure leaves the
    name unregistered, which an engine sees as an unavailable spec rather than as
    a crash.
    """
    try:
        OffloadingSpecFactory.register_spec(
            SPEC_NAME,
            "external_primary_tier.client.spec",
            SPEC_NAME,
        )
    except ValueError:
        # A registry entry already names this spec: another process, or another
        # call in this one, registered it first, and both name the same class.
        logger.debug("%s is already registered", SPEC_NAME)
    except Exception:
        logger.exception("Could not register the %s offloading spec", SPEC_NAME)
    else:
        logger.info("Registered the %s offloading spec", SPEC_NAME)
