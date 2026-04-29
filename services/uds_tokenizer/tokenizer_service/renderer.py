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

"""Renderer service wrapping vLLM's OpenAIServingRender for chat completion preprocessing."""

import logging
import threading

from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.plugins.io_processors import get_io_processor
from vllm.renderers import renderer_from_config


class RendererError(Exception):
    pass


class RendererService:
    """Wraps OpenAIServingRender for CPU-only chat preprocessing.

    Lazily loads one OpenAIServingRender per model; thread-safe.
    """

    def __init__(self):
        self._renderers: dict = {}  # model_name -> OpenAIServingRender
        self._lock = threading.Lock()

    def load_renderer(self, model_name: str, chat_template: str | None = None) -> bool:
        """Load (or no-op if already loaded) a renderer for model_name."""
        with self._lock:
            if model_name in self._renderers:
                return True

            try:
                self._renderers[model_name] = self._build_serving_render(
                    model_name, chat_template
                )
                logging.info("Renderer loaded for model: %s", model_name)
                return True
            except Exception as e:
                logging.error(
                    "Failed to load renderer for model %s: %s",
                    model_name,
                    e,
                    exc_info=True,
                )
                return False

    def _build_serving_render(self, model_name: str, chat_template: str | None):
        engine_args = AsyncEngineArgs(model=model_name, trust_remote_code=True)
        model_config = engine_args.create_model_config()
        vllm_config = VllmConfig(
            model_config=model_config, device_config=DeviceConfig(device="cpu")
        )
        renderer = renderer_from_config(vllm_config)
        io_processor = get_io_processor(vllm_config, renderer)

        model_registry = OpenAIModelRegistry(
            model_config=model_config,
            base_model_paths=[BaseModelPath(name=model_name, model_path=model_name)],
        )

        return OpenAIServingRender(
            model_config=model_config,
            renderer=renderer,
            io_processor=io_processor,
            model_registry=model_registry,
            request_logger=None,
            chat_template=chat_template,
            chat_template_content_format="auto",
            enable_auto_tools=True,
        )

    def _get_renderer(self, model_name: str):
        if not self.load_renderer(model_name):
            raise RendererError(f"Failed to load renderer for model: {model_name}")
        return self._renderers[model_name]

    async def render_chat(self, request: ChatCompletionRequest, model_name: str):
        """Render an OpenAI chat completion request, returning a vLLM GenerateRequest."""
        serving_render = self._get_renderer(model_name)
        result = await serving_render.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            raise RendererError(f"Render failed: {result.error.message}")
        return result

    async def render_completion(self, request: CompletionRequest, model_name: str):
        """Render an OpenAI completion request, returning a list of vLLM GenerateRequests."""
        serving_render = self._get_renderer(model_name)
        result = await serving_render.render_completion_request(request)
        if isinstance(result, ErrorResponse):
            raise RendererError(f"Render failed: {result.error.message}")
        return result
