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

"""
Direct gRPC server startup script for tokenizer service optimized for multi-core usage.
"""

import asyncio
import os
import json
import logging
import threading
import time
import signal
import sys

from aiohttp import web
from tokenizer_service.tokenizer import TokenizerService, TokenizerConfig
from tokenizer_grpc_service import create_grpc_server
from utils.thread_pool_utils import get_thread_pool

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
)

# Unix Domain Socket path
UDS_SOCKET_PATH = "/tmp/tokenizer/tokenizer-uds.socket"
# TCP probe port
PROBE_PORT = int(os.getenv("PROBE_PORT", 8082))  # use 8082 for probing to avoid conflicts

# Global variables for server control and configuration
grpc_server = None
probe_runner = None
probe_site = None
probe_loop = None  # Store the probe event loop for later use
probe_started_event = threading.Event()  # Event to signal when probe server has started
current_config = None
tokenizer_service = None
tokenizer_ready = False
shutdown_event = threading.Event()  # Event to signal shutdown

def _install_signal_handlers():
    """Install signal handlers for graceful shutdown"""
    def _signal_handler(signum, frame):
        logging.info(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


def initialize_tokenizer():
    """Initialize the tokenizer and set the ready flag"""
    global tokenizer_service, current_config, tokenizer_ready
    try:
        # Parse ADD_SPECIAL_TOKENS environment variable
        add_special_tokens_env = os.getenv("ADD_SPECIAL_TOKENS")
        if add_special_tokens_env is None or add_special_tokens_env.lower() == "none":
            add_special_tokens = None  # Use tokenizer's default behavior
        else:
            add_special_tokens = add_special_tokens_env.lower() == "true"

        current_config = TokenizerConfig(
            model=os.getenv("MODEL", "Qwen/Qwen3-0.6B"),
            add_special_tokens=add_special_tokens,
            enable_thinking=os.getenv("ENABLE_THINKING", "false").lower() == "true",
            add_generation_prompt=os.getenv("ADD_GENERATION_PROMPT", "true").lower() == "true"
        )
        tokenizer_service = TokenizerService(current_config)
        tokenizer_ready = True
        logging.info("Tokenizer initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer: {e}")
        raise


async def health_handler(request):
    """Health check endpoint"""
    global tokenizer_ready
    if not tokenizer_ready:
        return web.json_response({
            "status": "unhealthy",
            "service": "tokenizer-service",
            "reason": "tokenizer not ready",
            "timestamp": time.time()
        }, status=503)

    return web.json_response({
        "status": "healthy",
        "service": "tokenizer-service",
        "timestamp": time.time()
    })


async def config_handler(request):
    """Get current configuration"""
    global current_config
    if current_config is None:
        return web.json_response({"error": "Tokenizer not initialized"}, status=500)

    config_dict = {
        "model": current_config.model,
        "add_special_tokens": current_config.add_special_tokens,
        "enable_thinking": current_config.enable_thinking,
        "add_generation_prompt": current_config.add_generation_prompt
    }
    return web.json_response(config_dict)


async def update_config_handler(request):
    """Update configuration (hot reload)"""
    global tokenizer_service, current_config, tokenizer_ready
    try:
        body = await request.read()
        new_config_data = json.loads(body.decode('utf-8'))

        updated_config = TokenizerConfig(
            model=new_config_data.get("model", current_config.model),
            add_special_tokens=new_config_data.get("add_special_tokens", current_config.add_special_tokens),
            enable_thinking=new_config_data.get("enable_thinking", current_config.enable_thinking),
            add_generation_prompt=new_config_data.get("add_generation_prompt", current_config.add_generation_prompt)
        )

        tokenizer_ready = False

        # Reinitialize tokenizer service
        try:
            tokenizer_service = TokenizerService(updated_config)
            current_config = updated_config
            tokenizer_ready = True
            logging.info(f"Configuration updated: {new_config_data}")
            return web.json_response({
                "status": "success",
                "message": "Configuration updated successfully"
            })
        except Exception as e:
            # If initialization fails, restore previous configuration
            tokenizer_ready = True
            logging.error(f"Failed to initialize tokenizer with new config: {e}", exc_info=True)
            return web.json_response({
                "status": "error",
                "message": f"Failed to initialize tokenizer with new config: {e}"
            }, status=500)

    except Exception as e:
        logging.error(f"Config update error: {e}", exc_info=True)
        return web.json_response({
            "status": "error",
            "message": f"Config update failed: {e}"
        }, status=500)


def create_probe_app():
    """Create aiohttp application for probes and config"""
    app = web.Application()
    app.router.add_get('/health', health_handler)
    app.router.add_get('/config', config_handler)
    app.router.add_post('/config', update_config_handler)
    return app

# Import at the top of start_probe_server_in_background function
def start_probe_server_in_background():
    """Start the probe server in a background thread"""
    global probe_runner, probe_site, probe_loop

    # Create probe application (TCP socket) for health checks
    probe_app = create_probe_app()

    # Create a single event loop for the probe server
    probe_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(probe_loop)

    async def _start_probe_server():
        """Internal function to start the probe server"""
        global probe_runner, probe_site

        probe_runner = web.AppRunner(probe_app)
        await probe_runner.setup()

        probe_site = web.TCPSite(probe_runner, "0.0.0.0", PROBE_PORT)
        await probe_site.start()
        logging.info(f"Probe server started on port {PROBE_PORT}")

        # Signal that the probe server has started
        probe_started_event.set()

        # Wait for shutdown event using asyncio instead of blocking thread wait
        try:
            # Use asyncio Event for async waiting, not blocking wait
            while not shutdown_event.is_set():
                await asyncio.sleep(0.1)  # Small delay to prevent busy-waiting
        except asyncio.CancelledError:
            logging.info("Probe server task was cancelled")
            raise

    # Run the probe server in a background thread
    def run_probe_server():
        asyncio.set_event_loop(probe_loop)
        try:
            probe_loop.run_until_complete(_start_probe_server())
        except Exception as e:
            logging.error(f"Error starting probe server: {e}")
            raise
        finally:
            # Clean up the event loop resources only if it's not already closed
            if not probe_loop.is_closed():
                probe_loop.close()

    probe_thread = threading.Thread(target=run_probe_server, daemon=True)
    probe_thread.start()


def run_server():
    """Run the synchronous gRPC server with background probe server"""
    global tokenizer_service, grpc_server

    # Initialize tokenizer
    try:
        initialize_tokenizer()
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer, exiting: {e}")
        return

    # Remove old socket file if it exists
    if os.path.exists(UDS_SOCKET_PATH):
        os.remove(UDS_SOCKET_PATH)

    # Create dedicated directory and set permissions
    os.makedirs(os.path.dirname(UDS_SOCKET_PATH), mode=0o700, exist_ok=True)

    thread_pool = get_thread_pool()
    grpc_server = create_grpc_server(tokenizer_service, UDS_SOCKET_PATH, thread_pool)
    grpc_server.start()
    logging.info(f"Synchronous gRPC server started on {UDS_SOCKET_PATH}")

    # Start probe server in background
    start_probe_server_in_background()
    logging.info(f"Starting probe server on port {PROBE_PORT}")

    # Install signal handlers
    _install_signal_handlers()
    logging.info("Server started.")

    shutdown_event.wait()
    logging.info("Shutdown signal received, initiating graceful shutdown...")

    # Stop gRPC server gracefully
    grace_period = float(os.getenv("GRACE_PERIOD_SECONDS", "30.0"))
    logging.info(f"Stopping gRPC server with grace period of {grace_period}s...")
    stop_future = grpc_server.stop(grace=grace_period)
    stop_future.wait(timeout=grace_period + 5.0)

    logging.info("gRPC server stopped.")

    # Shutdown probe server
    shutdown_probe_server()


def shutdown_probe_server():
    """Shutdown probe server gracefully"""
    global probe_runner, probe_site, probe_loop, probe_started_event
    try:
        if probe_runner:
            logging.info("Shutting down probe server...")

            # Signal the shutdown event to stop the probe server loop
            shutdown_event.set()

            # Wait for probe server to have started before attempting to shut down
            # This prevents race condition where shutdown happens before server fully starts
            # Using timeout to avoid hanging indefinitely if probe server never starts
            probe_started_event.wait(timeout=30.0)

            async def stop_probe_server():
                try:
                    if probe_site:
                        await probe_site.stop()
                    if probe_runner:
                        await probe_runner.cleanup()
                    logging.info("Probe server stopped")
                except Exception as e:
                    logging.error(f"Error stopping probe server: {e}")

            # Run cleanup using the existing event loop if it's available and not closed
            if probe_loop and not probe_loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(stop_probe_server(), probe_loop)
                try:
                    # Wait for cleanup to complete (with timeout)
                    future.result(timeout=10.0)  # 10 second timeout
                except Exception as e:
                    logging.error(f"Error during probe server cleanup: {e}")

        # Reset the probe started event for potential future use
        probe_started_event.clear()

    except Exception as e:
        logging.error(f"Error during probe server shutdown: {e}")


if __name__ == "__main__":
    try:
        run_server()
        logging.info("Server shutdown complete")
        sys.exit(0)
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)