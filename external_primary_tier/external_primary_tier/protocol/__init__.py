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

"""The protocol between engine processes and the External Primary Tier."""

from external_primary_tier.protocol.codec import Codec, client_codec, tier_codec
from external_primary_tier.protocol.messages import (
    PROTOCOL_VERSION,
    Attach,
    AttachOk,
    CompleteLoad,
    CompleteStore,
    Draining,
    FromTier,
    GetStats,
    Leave,
    LoadSpec,
    Lookup,
    LookupOk,
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
    ToTier,
    UnknownClient,
)
from external_primary_tier.protocol.region import AnyRegion, RegionDescriptor, ShmFileRegion

__all__ = [
    "PROTOCOL_VERSION",
    "AnyRegion",
    "Attach",
    "AttachOk",
    "Codec",
    "CompleteLoad",
    "CompleteStore",
    "Draining",
    "FromTier",
    "GetStats",
    "Leave",
    "LoadSpec",
    "Lookup",
    "LookupOk",
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
    "ToTier",
    "UnknownClient",
    "client_codec",
    "tier_codec",
]
