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

"""Encoding of protocol messages, with vLLM's msgpack encoder and decoder.

vLLM's ``MsgpackDecoder`` hands its type to msgspec's decoder, so decoding a
union of tagged Structs returns the class each message's tag names.
"""

from collections.abc import Sequence
from typing import Any

from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, bytestr

from external_primary_tier.protocol.messages import FromTier, ToTier


class Codec:
    """Encodes the messages one side sends and decodes the ones it receives."""

    def __init__(self, incoming: Any) -> None:
        self._encoder = MsgpackEncoder()
        self._decoder = MsgpackDecoder(incoming)

    def encode(self, message: Any) -> Sequence[bytestr]:
        """The frames to send. Protocol messages carry no tensors, so one frame."""
        return self._encoder.encode(message)

    def decode(self, frames: bytestr | Sequence[bytestr]) -> Any:
        """Raises msgspec.ValidationError for a message this side does not receive."""
        return self._decoder.decode(frames)


def tier_codec() -> Codec:
    """For the External Primary Tier: decodes what engines send."""
    return Codec(ToTier)


def client_codec() -> Codec:
    """For engine processes: decodes what the External Primary Tier sends."""
    return Codec(FromTier)
