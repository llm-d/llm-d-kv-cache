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
import logging
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
    parser.add_argument(
        "--cache-policy",
        default="lru",
        help="vLLM's cache policy for the pool: lru or arc.",
    )
    parser.add_argument(
        "--cache-policy-module-path",
        default=None,
        help="Module to load --cache-policy from when it is not one of vLLM's built-in policies, "
        "e.g. my_package.my_policy.",
    )
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=30.0,
        help="How long a pin or write lasts before it is released; far longer than the longest copy.",
    )
    parser.add_argument(
        "--fence-seconds",
        type=float,
        default=10.0,
        help="After a restart, how long the previous instance's leases stay, or, after a crash, how long no "
        "chunk is handed out: longer than the longest copy plus the time an engine takes to see a request "
        "time out.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Level of the tier's log.",
    )
    args = parser.parse_args(argv)

    # vLLM configures only its own "vllm" logger, and the tier's loggers are not under it.
    logging.basicConfig(
        level=args.log_level,
        format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    )

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    tier = ExternalPrimaryTier(
        args.address,
        args.region_dir,
        args.pool_bytes,
        cache_policy=args.cache_policy,
        cache_policy_module_path=args.cache_policy_module_path,
        lease_seconds=args.lease_seconds,
        fence_seconds=args.fence_seconds,
    )
    try:
        tier.serve(stop)
    finally:
        tier.close()


if __name__ == "__main__":
    main()
