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

"""Version 1 of the protocol between engine processes and the pool server."""

from kvpoold.api.v1.compat import COMPAT_ID_BYTES, CompatDescriptor, LayoutId, NamespaceId
from kvpoold.api.v1.messages import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompleteLoad,
    CompleteStore,
    Draining,
    FromServer,
    GetStats,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
    NoSpace,
    Notice,
    OffloadKey,
    PrepareLoad,
    PrepareStore,
    Refused,
    ReleaseHits,
    Request,
    RequestFinished,
    Stats,
    StoreSpec,
    ToServer,
    UnknownClient,
)
from kvpoold.api.v1.region import RegionDescriptor, ShmFileRegion

__all__ = [
    "COMPAT_ID_BYTES",
    "PROTOCOL_VERSION",
    "Attach",
    "AttachOk",
    "CompatDescriptor",
    "CompleteLoad",
    "CompleteStore",
    "Draining",
    "FromServer",
    "GetStats",
    "LayoutId",
    "Leave",
    "LoadSpec",
    "Lookup",
    "LookupOk",
    "NamespaceId",
    "NoSpace",
    "Notice",
    "OffloadKey",
    "PrepareLoad",
    "PrepareStore",
    "Refused",
    "RegionDescriptor",
    "ReleaseHits",
    "Request",
    "RequestFinished",
    "ShmFileRegion",
    "Stats",
    "StoreSpec",
    "ToServer",
    "UnknownClient",
]
