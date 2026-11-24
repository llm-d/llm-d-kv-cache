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

"""``kvpoold``: run the pool server."""

import argparse
import logging
import signal
import threading

from vllm.config.kv_events import KVEventsConfig

from kvpoold.server.app import Server
from kvpoold.server.config import ServerConfig


def parse_args(argv: list[str] | None = None) -> tuple[ServerConfig, str]:
    """The server's config, and the log level, from the command line."""
    parser = argparse.ArgumentParser(
        prog="kvpoold",
        description="The per-node pool server: one CPU KV-cache pool shared by the node's vLLM engines.",
    )
    parser.add_argument("--pool-bytes", type=int, required=True, help="Size of the shared CPU pool.")
    parser.add_argument(
        "--address",
        required=True,
        help="ZMQ endpoint to bind, e.g. ipc:///host-shm/kvpoold.sock",
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
        "--kv-events-endpoint",
        required=True,
        help="ZMQ endpoint to publish the pool's KV events on, as vLLM's --kv-events-config endpoint "
        "(it is bound when it has a '*' or is ipc://, and connected to otherwise), "
        "e.g. tcp://*:5557.",
    )
    parser.add_argument(
        "--kv-events-topic",
        required=True,
        help="The topic every KV events batch is published on, which the deployment sets as it does for "
        "an engine, e.g. kv@${NODE_NAME}@<model>. Not empty.",
    )
    parser.add_argument(
        "--kv-events-replay-endpoint",
        default=None,
        help="ZMQ endpoint to bind for subscribers to ask for the batches they missed, "
        "e.g. tcp://*:5558. Without it, no replay.",
    )
    parser.add_argument(
        "--kv-events-buffer-steps",
        type=int,
        default=KVEventsConfig.buffer_steps,
        help="How many past batches are kept for replay.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Level of the server's log.",
    )
    args = parser.parse_args(argv)
    if not args.kv_events_endpoint:
        parser.error("--kv-events-endpoint must not be empty")
    if not args.kv_events_topic:
        parser.error("--kv-events-topic must not be empty")

    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        endpoint=args.kv_events_endpoint,
        replay_endpoint=args.kv_events_replay_endpoint,
        buffer_steps=args.kv_events_buffer_steps,
        topic=args.kv_events_topic,
    )
    config = ServerConfig(
        address=args.address,
        region_dir=args.region_dir,
        pool_bytes=args.pool_bytes,
        kv_events_config=kv_events_config,
        cache_policy=args.cache_policy,
        cache_policy_module_path=args.cache_policy_module_path,
        lease_seconds=args.lease_seconds,
        fence_seconds=args.fence_seconds,
    )
    return config, args.log_level


def main(argv: list[str] | None = None) -> None:
    config, log_level = parse_args(argv)

    # The server's loggers are under "kvpoold", which propagates to the root logger.
    logging.basicConfig(
        level=log_level,
        format="%(levelname)s %(asctime)s [%(name)s] %(message)s",
    )

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    server = Server(config)
    try:
        server.serve(stop)
    finally:
        server.close()


if __name__ == "__main__":
    main()
