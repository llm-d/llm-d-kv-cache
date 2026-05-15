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

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

PATCHED_FLAG = "llmd_fs_patched"


def install_offload_metric_suffix_patch() -> None:
    """
    Patch vLLM OffloadPromMetrics so each OffloadingConnector instance
    registers Prometheus metrics under a spec_name-derived suffix
    (vllm:kv_offload_total_bytes_sharedstorage, ..._cpu, ...).

    Without this, MultiConnector + two OffloadingConnector children hit
    "Duplicated timeseries in CollectorRegistry" because both instances
    register the same fixed metric names. Idempotent; safe to call
    multiple times.

    TODO: remove once vLLM upstream applies an equivalent spec_name
    suffix (or other disambiguation) to the OffloadPromMetrics names.
    """
    try:
        from prometheus_client import Counter, Histogram
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            offloading_connector as oc,
        )
    except Exception as exc:
        log.debug("skipping OffloadPromMetrics patch: %s", exc)
        return

    # Idempotency guard: mark the class after patching; safe to call many times.
    if getattr(oc.OffloadPromMetrics, PATCHED_FLAG, False):
        return

    # Keep a handle to the real __init__; the patched version delegates to it.
    orig_init = oc.OffloadPromMetrics.__init__

    def patched_init(
        self, vllm_config, metric_types, labelnames, per_engine_labelvalues
    ):
        # Per-connector suffix derived from spec_name.
        # e.g. "SharedStorageOffloadingSpec" -> "sharedstorage",
        #      "CPUOffloadingSpec"           -> "cpu".
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        suffix = (
            str(extra.get("spec_name", "default")).replace("OffloadingSpec", "").lower()
            or "default"
        )

        # wrap(cls) returns a drop-in replacement that tries the upstream
        # name first; only when Prometheus reports a duplicate (the second
        # OffloadPromMetrics under MultiConnector) does it retry with a
        # spec_name suffix. Single-connector deployments keep the original
        # vllm:kv_offload_total_bytes / _time / _size names unchanged.
        def wrap(cls):
            def factory(**kwargs):
                try:
                    return cls(**kwargs)
                except ValueError as e:
                    if "Duplicated timeseries" in str(e) and "name" in kwargs:
                        kwargs["name"] = f"{kwargs['name']}_{suffix}"
                        return cls(**kwargs)
                    raise

            return factory

        # Copy before mutating — metric_types may be shared across instances.
        patched = dict(metric_types)
        if Counter in patched:
            patched[Counter] = wrap(patched[Counter])
        if Histogram in patched:
            patched[Histogram] = wrap(patched[Histogram])

        # Delegate to the real __init__ with the wrapped factories.
        orig_init(self, vllm_config, patched, labelnames, per_engine_labelvalues)

    # Monkey-patch: replace the class method and mark it patched.
    oc.OffloadPromMetrics.__init__ = patched_init
    setattr(oc.OffloadPromMetrics, PATCHED_FLAG, True)
    log.info("installed OffloadPromMetrics spec-suffix patch")
