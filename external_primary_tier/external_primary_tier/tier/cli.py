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

"""``external-primary-tier``: run the External Primary Tier."""

import argparse
import signal
import threading

from external_primary_tier.tier.server import ExternalPrimaryTier


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="external-primary-tier",
        description="The per-node External Primary Tier: one CPU KV-cache pool shared by the node's vLLM engines.",
    )
    parser.add_argument("--pool-bytes", type=int, required=True, help="Size of the shared CPU pool.")
    parser.add_argument(
        "--address",
        required=True,
        help="ZMQ endpoint to bind, e.g. ipc:///host-shm/external-primary-tier.sock",
    )
    parser.add_argument(
        "--region-dir",
        required=True,
        help="Directory for the region's files, e.g. /host-shm",
    )
    args = parser.parse_args(argv)

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    tier = ExternalPrimaryTier(args.address, args.region_dir, args.pool_bytes)
    try:
        tier.serve(stop)
    finally:
        tier.close()


if __name__ == "__main__":
    main()
