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

import asyncio
import json
import os

import zmq
import zmq.asyncio
from msgspec.msgpack import Decoder

os.environ["PYTHONHASHSEED"] = "42"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from huggingface_hub import snapshot_download
from vllm import LLM
from vllm.config.kv_events import KVEventsConfig
from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.lora.request import LoRARequest

MODEL_NAME = "Qwen/Qwen3-0.6B"  # Small Qwen model that supports LoRA
ZMQ_TOPIC = f"kv@localhost@{MODEL_NAME}"
ZMQ_PORT = 5557
LORA_MODEL = "charent/self_cognition_Alice"  # Working LoRA adapter
LORA_NAME = "Alice"


def patch_engine_args():
    # Force enable prefix caching by patching EngineArgs property
    # NOTE: vLLM disables prefix caching on non-x86_64 architectures because
    # CPU-based paged attention is only implemented for x86_64. Other architectures
    # skip KV cache functionality entirely.
    def always_true_prefix_caching(self):
        # EngineArgs.enable_prefix_caching: Always returning True
        return True

    def set_prefix_caching(self, value):
        # Ignoring attempt to set prefix_caching, keeping True
        pass

    # Replace the enable_prefix_caching attribute with our property
    EngineArgs.enable_prefix_caching = property(
        fget=always_true_prefix_caching, fset=set_prefix_caching
    )


def create_llm():
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{ZMQ_PORT}",
        topic=ZMQ_TOPIC,
    )

    llm = LLM(
        model=MODEL_NAME,
        enable_prefix_caching=True,
        dtype="float16",
        enforce_eager=True,
        disable_hybrid_kv_cache_manager=True,
        kv_events_config=kv_events_config,
        block_size=16,
        prefix_caching_hash_algo="sha256_cbor_64bit",
        enable_lora=True,
        max_model_len=4096,
    )
    return llm


async def listen_for_kv_event() -> list[BlockStored | BlockRemoved | AllBlocksCleared]:
    """
    Listens for KV cache events using a basic ZMQ SUB socket.
    Removes sequence number checking and replay logic.
    """
    decoder = Decoder(type=KVEventBatch)
    context = zmq.asyncio.Context()
    sub = context.socket(zmq.SUB)
    sub.connect(f"tcp://localhost:{ZMQ_PORT}")
    topic = ZMQ_TOPIC
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)

    print("[ZMQ] Listener started and waiting for events on topic:", topic)

    events = []

    # We wait for the FIRST message (expected to be [topic, seq, payload])
    try:
        # Use a timeout to prevent an infinite block if the publisher fails
        # The inference run should complete well within this time.
        _, seq_bytes, payload = await asyncio.wait_for(
            sub.recv_multipart(), timeout=60.0
        )

        # Decode and store the events
        event_batch = decoder.decode(payload)
        events.extend(event_batch.events)
        print(f"[ZMQ] Received {len(events)} events in the first batch.")

    except asyncio.TimeoutError:
        print("[ZMQ] Timeout (60s) reached while waiting for the first event batch.")
    except Exception as e:
        print(f"[ZMQ] Listener: An error occurred while receiving ZMQ message: {e}")

    # Clean up (This happens AFTER the task has finished waiting/receiving)
    sub.close()
    context.term()

    return events


async def main(with_lora: bool):
    # 0. Download LoRA adapter
    lora_path = snapshot_download(repo_id=LORA_MODEL)

    # 1. Start listening for events in background IMMEDIATELY
    # The listener runs in a task while the rest of main executes.
    event_task = asyncio.create_task(listen_for_kv_event())

    # Cause the event loop to switch to the listener task right away
    await asyncio.sleep(0)

    print("\n--- Starting LLM Initialization ---")
    patch_engine_args()

    # 2. Initialize the LLM (This takes time and runs while event_task is waiting)
    llm = create_llm()
    print("--- LLM Initialization Complete ---")

    prompt = """
    Anarchism is a political philosophy and movement that is sceptical of authority and rejects all involuntary, coercive forms of hierarchy. Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful. As a historically left-wing movement, placed on the farthest left of the political spectrum, it is usually described alongside communalism and libertarian Marxism as the libertarian wing (libertarian socialism) of the socialist movement, and has a strong historical association with anti-capitalism and socialism.
    Humans lived in societies without formal hierarchies long before the establishment of formal states, realms, or empires. With the rise of organised hierarchical bodies, scepticism toward authority also rose. Although traces of anarchist thought are found throughout history, modern anarchism emerged from the Enlightenment. During the latter half of the 19th and the first decades of the 20th century, the anarchist movement flourished in most parts of the world and had a significant role in workers' struggles for emancipation. Various anarchist schools of thought formed during this period. Anarchists have taken part in several revolutions, most notably in the Paris Commune, the Russian Civil War and the Spanish Civil War, whose end marked the end of the classical era of anarchism. In the last decades of the 20th and into the 21st century, the anarchist movement has been resurgent once more.
    Anarchism employs a diversity of tactics in order to meet its ideal ends which can be broadly separated into revolutionary and evolutionary tactics; there is significant overlap between the two, which are merely descriptive. Revolutionary tactics aim to bring down authority and state, having taken a violent turn in the past, while evolutionary tactics aim to prefigure what an anarchist society would be like. Anarchist thought, criticism, and praxis have played a part in diverse areas of human society. Criticism of anarchism include claims that it is internally inconsistent, violent, or utopian.
    """.strip()  # noqa: E501

    if with_lora:
        print("\n--- Request With Alice LoRA ---")
        llm.generate([prompt], lora_request=LoRARequest(LORA_NAME, 1, lora_path))
    else:
        print("\n--- Request to Base model (no LoRA) ---")
        llm.generate([prompt])

    print("--- Inference Complete. Waiting for Listener Task ---")

    # 4. Wait for and get the results from the event task
    events = await event_task

    print(f"\nReceived {len(events)} KV cache events:")
    for event in events:
        print(f"  - {event}")

    events_json = {
        "prompt": prompt,
        "model_name": MODEL_NAME,
        "lora_path": LORA_MODEL,
        "lora_name": events[0].lora_name,
        "event_type": events[0].__class__.__name__,
        "block_hashes": events[0].block_hashes,
        "parent_block_hash": events[0].parent_block_hash,
        "token_ids": events[0].token_ids,
        "block_size": events[0].block_size,
        "medium": events[0].medium,
        "hash_seed": os.environ["PYTHONHASHSEED"],
    }

    fname = "kv_event_lora.json" if with_lora else "kv_event_base.json"
    with open(fname, "w") as f:
        json.dump(events_json, f, indent=4)
    print("\nKV events saved to kv_events_lora.json")


if __name__ == "__main__":
    asyncio.run(main(with_lora=True))
