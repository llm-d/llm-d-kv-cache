#!/usr/bin/env python3
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

"""Async gRPC server startup script for tokenizer service."""

import asyncio
import os
import logging
import time
import signal

from aiohttp import web
from tokenizer_service.tokenizer import TokenizerService
from tokenizer_service.renderer import RendererService
from tokenizer_grpc_service import create_grpc_server

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)

# Unix Domain Socket path
UDS_SOCKET_PATH = "/tmp/tokenizer/tokenizer-uds.socket"
# TCP probe port
PROBE_PORT = int(
    os.getenv("PROBE_PORT", 8082)
)  # use 8082 for probing to avoid conflicts
# TCP gRPC port (FOR TESTING ONLY - do not use in production)
# If not set, only UDS is used (production default)
GRPC_PORT = os.getenv("GRPC_PORT", "")  # e.g., "50051" for tests
GRACE_PERIOD_SECONDS = float(os.getenv("GRACE_PERIOD_SECONDS", "30.0"))


async def run_server():
    ready = False

    tokenizer_service = TokenizerService()
    renderer_service = RendererService()

    if os.path.exists(UDS_SOCKET_PATH):
        os.remove(UDS_SOCKET_PATH)
    os.makedirs(os.path.dirname(UDS_SOCKET_PATH), mode=0o700, exist_ok=True)

    server = create_grpc_server(
        tokenizer_service, UDS_SOCKET_PATH, renderer_service, GRPC_PORT
    )
    await server.start()
    ready = True
    logging.info(
        f"gRPC server started on {UDS_SOCKET_PATH}"
        + (f" and TCP port {GRPC_PORT}" if GRPC_PORT else "")
    )

    # Probe server
    def _health(r):
        if not ready:
            return web.json_response(
                {
                    "status": "unhealthy",
                    "reason": "not ready",
                    "timestamp": time.time(),
                },
                status=503,
            )
        return web.json_response(
            {
                "status": "healthy",
                "service": "tokenizer-service",
                "timestamp": time.time(),
            }
        )

    app = web.Application()
    app.router.add_get("/healthz", _health)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", PROBE_PORT).start()
    logging.info(f"Probe server started on port {PROBE_PORT}")

    # Shutdown on SIGTERM/SIGINT
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _on_signal():
        logging.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _on_signal)

    logging.info("Server started.")
    await shutdown_event.wait()
    ready = False
    logging.info(f"Stopping gRPC server (grace={GRACE_PERIOD_SECONDS}s)...")
    await server.stop(grace=GRACE_PERIOD_SECONDS)
    await runner.cleanup()
    logging.info("Server shutdown complete.")


if __name__ == "__main__":
    asyncio.run(run_server())
