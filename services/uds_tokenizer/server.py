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
"""Web server for tokenizer service."""

import asyncio
import json
import logging
import os
import signal

from aiohttp import web
from tokenizer_service.tokenizer import TokenizerConfig, TokenizerService

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s')

# Try to use uvloop for better performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logging.info("Using uvloop for better performance")
except ImportError:
    logging.warning("uvloop not available, using default asyncio event loop")

# Unix Domain Socket path
UDS_SOCKET_PATH = "/tmp/tokenizer/tokenizer-uds.socket"
# TCP probe port
PROBE_PORT = int(os.getenv("PROBE_PORT", 8080))

# Global variables for server control and configuration
server_runner = None
server_site = None
probe_runner = None
probe_site = None
shutdown_event = None
current_config = None
tokenizer_service = None
tokenizer_ready = False


def initialize_tokenizer():
    """Initialize the tokenizer and set the ready flag"""
    global tokenizer_service, current_config, tokenizer_ready
    try:
        # Parse ADD_SPECIAL_TOKENS environment variable
        add_special_tokens_env = os.getenv("ADD_SPECIAL_TOKENS")
        if add_special_tokens_env is None or add_special_tokens_env.lower(
        ) == "none":
            add_special_tokens = None  # Use tokenizer's default behavior
        else:
            add_special_tokens = add_special_tokens_env.lower() == "true"

        current_config = TokenizerConfig(
            model=os.getenv("MODEL", "Qwen/Qwen3-0.6B"),
            add_special_tokens=add_special_tokens,
            enable_thinking=os.getenv("ENABLE_THINKING",
                                      "false").lower() == "true",
            add_generation_prompt=os.getenv("ADD_GENERATION_PROMPT",
                                            "true").lower() == "true")
        tokenizer_service = TokenizerService(current_config)
        tokenizer_ready = True
        logging.info("Tokenizer initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize tokenizer: {e}")
        raise


async def template_handler(request):
    """Handle chat template requests"""
    logging.info("Handling chat template request")
    try:
        body = await request.read()

        try:
            messages = json.loads(body.decode('utf-8'))
        except UnicodeDecodeError as e:
            logging.error(f"Invalid UTF-8 encoding: {e}")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Invalid UTF-8 encoding: {e}"
                },
                status=400)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON: {e}")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Invalid JSON: {e}"
                },
                status=400)

        prompt = tokenizer_service.apply_template(messages)
        prompt = prompt[0]
        logging.info(f"Generated prompt: {prompt[:100]}...")
        return web.Response(text=prompt, content_type='text/plain')

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "message": f"Processing failed: {e}"
            },
            status=500)


async def tokenize_handler(request):
    """Handle tokenization requests"""
    logging.info("Handling tokenize request")
    try:
        body = await request.read()

        prompt = body.decode('utf-8')
        logging.info(f"Prompt to tokenize: {prompt[:100]}...")

        loop = asyncio.get_running_loop()
        batch_encoding = await loop.run_in_executor(
            None, tokenizer_service.tokenize_and_process, prompt)
        serializable_data = {
            key: value.tolist() if hasattr(value, "tolist") else value
            for key, value in batch_encoding.items()
        }
        response = json.dumps(serializable_data)
        return web.Response(text=response, content_type='application/json')

    except Exception as e:
        logging.error(f"Processing error: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "message": f"Processing failed: {e}"
            },
            status=500)


async def health_handler(request):
    """Health check endpoint"""
    if not tokenizer_ready:
        return web.json_response(
            {
                "status": "unhealthy",
                "service": "tokenizer-service",
                "reason": "tokenizer not ready",
                "timestamp": asyncio.get_event_loop().time()
            },
            status=503)

    return web.json_response({
        "status": "healthy",
        "service": "tokenizer-service",
        "timestamp": asyncio.get_event_loop().time()
    })


async def config_handler(request):
    """Get current configuration"""
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
            add_special_tokens=new_config_data.get(
                "add_special_tokens", current_config.add_special_tokens),
            enable_thinking=new_config_data.get(
                "enable_thinking", current_config.enable_thinking),
            add_generation_prompt=new_config_data.get(
                "add_generation_prompt", current_config.add_generation_prompt))

        tokenizer_ready = False

        # Reinitialize tokenizer service
        try:
            tokenizer_service = TokenizerService(updated_config)
            current_config = updated_config
            tokenizer_ready = True
            logging.info(f"Configuration updated: {new_config_data}")
            return web.json_response({
                "status":
                "success",
                "message":
                "Configuration updated successfully"
            })
        except Exception as e:
            # If initialization fails, restore previous configuration
            tokenizer_ready = True
            logging.error(
                f"Failed to initialize tokenizer with new config: {e}",
                exc_info=True)
            return web.json_response(
                {
                    "status":
                    "error",
                    "message":
                    f"Failed to initialize tokenizer with new config: {e}"
                },
                status=500)

    except Exception as e:
        logging.error(f"Config update error: {e}", exc_info=True)
        return web.json_response(
            {
                "status": "error",
                "message": f"Config update failed: {e}"
            },
            status=500)


def create_app():
    """Create aiohttp application"""
    app = web.Application()
    app.router.add_post('/chat-template', template_handler)
    app.router.add_post('/tokenize', tokenize_handler)
    app.router.add_get('/health', health_handler)
    app.router.add_get('/config', config_handler)
    app.router.add_post('/config', update_config_handler)
    return app


def create_probe_app():
    """Create aiohttp application for probes"""
    app = web.Application()
    app.router.add_get('/health', health_handler)
    return app


async def shutdown_handler(app):
    """Application shutdown handler"""
    logging.info("Shutting down application")


async def cleanup():
    """Clean up resources"""
    global server_runner, server_site, probe_runner, probe_site
    logging.info("Cleaning up resources")

    if probe_site:
        await probe_site.stop()
        logging.info("Probe site stopped")

    if probe_runner:
        await probe_runner.cleanup()
        logging.info("Probe runner cleaned up")

    if server_site:
        await server_site.stop()
        logging.info("Server site stopped")

    if server_runner:
        await server_runner.cleanup()
        logging.info("Server runner cleaned up")

    if os.path.exists(UDS_SOCKET_PATH):
        os.remove(UDS_SOCKET_PATH)
        logging.info(f"Socket file {UDS_SOCKET_PATH} removed")


async def run_server():
    """Run the server"""
    global server_runner, server_site, probe_runner, probe_site, shutdown_event

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

    # Create main application (UDS)
    app = create_app()
    app.on_shutdown.append(shutdown_handler)

    server_runner = web.AppRunner(app)
    await server_runner.setup()

    server_site = web.UnixSite(server_runner, UDS_SOCKET_PATH)

    # Create probe application (TCP socket)
    probe_app = create_probe_app()
    probe_runner = web.AppRunner(probe_app)
    await probe_runner.setup()

    probe_site = web.TCPSite(probe_runner, "0.0.0.0", PROBE_PORT)

    # Set up signal handling
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def signal_handler():
        logging.info("Received signal, initiating shutdown...")
        shutdown_event.set()

    loop.add_signal_handler(signal.SIGTERM, signal_handler)
    loop.add_signal_handler(signal.SIGINT, signal_handler)

    try:
        await server_site.start()
        await probe_site.start()
        print(f"Starting HTTP server on {UDS_SOCKET_PATH}")
        print(f"Starting probe server on port {PROBE_PORT}")
        await shutdown_event.wait()
        logging.info("Shutdown event received")
    except Exception as e:
        logging.error(f"Server error: {e}", exc_info=True)
    finally:
        await cleanup()


# Gunicorn entry point
async def create_app_for_gunicorn():
    """Create application for Gunicorn"""
    global tokenizer_service

    # Use a lock file to synchronize tokenizer initialization across workers
    lock_file_path = "/tmp/tokenizer_init.lock"

    if tokenizer_service is None:
        import fcntl

        # Ensure lock file exists
        open(lock_file_path, 'a').close()

        # Open lock file
        lock_file = open(lock_file_path, 'r+')

        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            logging.info("Acquired tokenizer initialization lock")

            if tokenizer_service is None:
                logging.info("Initializing tokenizer...")
                try:
                    initialize_tokenizer()
                except Exception as e:
                    logging.error(
                        f"Failed to initialize tokenizer in gunicorn mode: {e}"
                    )
                    raise
            else:
                logging.info("Tokenizer already initialized by another worker")

        finally:
            # Release the lock
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()

    return create_app()


if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
