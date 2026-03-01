// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package helper

import (
	"context"
	"fmt"
	"time"

	"github.com/llm-d/llm-d-kv-cache/examples/testdata"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/vmihailenco/msgpack/v5"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const topic = "kv@vllm-pod1@" + testdata.ModelName

func SimulateProduceEvent(ctx context.Context, publisher *Publisher) error {
	logger := log.FromContext(ctx)
	logger.Info("@@@ Simulating vLLM engine publishing BlockStored events...")
	medium := "GPU"

	// Create event in vLLM msgpack array format: [tag, hashes, parent, tokens, blockSize, loraID, medium, loraName]
	blockStoredEvent := []any{
		"BlockStored",         // Tag
		testdata.PromptHashes, // BlockHashes (already []uint64)
		nil,                   // ParentBlockHash
		[]uint32{1, 2, 3},     // TokenIds
		256,                   // BlockSize
		nil,                   // LoraID
		medium,                // Medium
		nil,                   // LoraName
	}

	//nolint // won't fail
	blockStoredPayload, _ := msgpack.Marshal(blockStoredEvent)

	// Create vLLM msgpack event batch in array format: [timestamp, [event1, event2, ...], data_parallel_rank]
	eventBatch := []any{
		float64(time.Now().UnixNano()) / 1e9, // Timestamp
		[][]byte{blockStoredPayload},         // Events array
		nil,                                  // DataParallelRank
	}

	if err := publisher.PublishEvent(ctx, topic, eventBatch); err != nil {
		return fmt.Errorf("failed to publish BlockStored event: %w", err)
	}
	logger.Info("@@@ Published BlockStored event", "topic", topic, "blocks", 3)

	// Give the subscriber a moment to connect
	time.Sleep(1 * time.Second)

	// Wait for events to be processed by the pool
	logger.Info("@@@ Waiting for events to be processed...")
	time.Sleep(3 * time.Second)

	return nil
}

func SimulateRemoveEvent(ctx context.Context, publisher *Publisher) error {
	logger := log.FromContext(ctx)
	logger.Info("@@@ Simulating vLLM engine removing some blocks...")

	// Create event in vLLM msgpack array format: [tag, hashes]
	blockRemovedEvent := []any{
		"BlockRemoved",
		[]uint64{testdata.PromptHashes[2], testdata.PromptHashes[3]},
	}

	//nolint // won't fail
	blockRemovedPayload, _ := msgpack.Marshal(blockRemovedEvent)

	// Create vLLM msgpack event batch in array format: [timestamp, [event1, event2, ...], data_parallel_rank]
	removeEventBatch := []any{
		float64(time.Now().UnixNano()) / 1e9,
		[][]byte{blockRemovedPayload},
		nil,
	}

	if err := publisher.PublishEvent(ctx, topic, removeEventBatch); err != nil {
		return fmt.Errorf("failed to publish BlockRemoved event: %w", err)
	}
	logger.Info("@@@ Published BlockRemoved event", "topic", topic, "blocks", 2)

	// Wait for removal events to be processed
	time.Sleep(3 * time.Second)

	return nil
}

func SetupEventsPool(ctx context.Context, kvBlockIndex kvblock.Index) (*kvevents.Pool, error) {
	logger := log.FromContext(ctx)

	cfg := kvevents.DefaultConfig()

	logger.Info("Creating events pool", "config", cfg)

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	if err != nil {
		return nil, err
	}

	pool := kvevents.NewPool(cfg, kvBlockIndex, tokenProcessor)

	return pool, nil
}
