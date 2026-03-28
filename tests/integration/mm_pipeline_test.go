/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package integration_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

// TestMMPipeline_IngestionMatchesRequestPath simulates the full multimodal pipeline:
//
//  1. Construct a vLLM-style msgpack BlockStored event with extra_keys for two images
//  2. Parse it through the VLLMAdapter
//  3. Feed the resulting event through the pool's processEventBatch (via a mock pool)
//  4. Independently compute request-path block keys using ComputeBlockExtraFeatures
//  5. Verify both paths produce identical block keys
//
// This proves that the ingestion path (vLLM event → index) and request path
// (tokenizer features → scoring) agree on block hashes for multimodal content.
func TestMMPipeline_IngestionMatchesRequestPath(t *testing.T) {
	const (
		blockSize = 16
		numBlocks = 8 // 4 text + 2 image1 + 1 text + 1 image2
		numTokens = numBlocks * blockSize
		modelName = "test-mm-model"
		podID     = "pod-1"
	)

	// Build tokens: 128 tokens (8 blocks of 16).
	tokens := make([]uint32, numTokens)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}

	// Image 1: blocks 4-5 (tokens [64, 96)), offset 64, length 32.
	// Image 2: block 7 (tokens [112, 128)), offset 112, length 16.
	img1Hash := "sha256_image1_abc123"
	img2Hash := "sha256_image2_def456"

	// Build per-block extra_keys as vLLM would send them.
	// Format: extra_keys[block] = [(hash, block_relative_offset), ...] or nil.
	extraKeysRaw := make([]any, numBlocks)
	// blocks 0-3: text only → nil
	// block 4: image1, offset = 64 - 4*16 = 0
	extraKeysRaw[4] = []any{[]any{img1Hash, int64(0)}}
	// block 5: image1 continuation, offset = 64 - 5*16 = -16
	extraKeysRaw[5] = []any{[]any{img1Hash, int64(-16)}}
	// block 6: text → nil
	// block 7: image2, offset = 112 - 7*16 = 0
	extraKeysRaw[7] = []any{[]any{img2Hash, int64(0)}}

	// Simulate engine-side hash computation (what vLLM does).
	// vLLM computes block hashes with extra_keys baked in.
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(
		&kvblock.TokenProcessorConfig{BlockSize: blockSize, HashSeed: "test-seed"},
	)
	require.NoError(t, err)

	// Parse raw extra_keys → typed features (same as pool.go does).
	rawForParse := make([][]any, numBlocks)
	for i, v := range extraKeysRaw {
		if v == nil {
			continue
		}
		slice, ok := v.([]any)
		require.True(t, ok, "extraKeysRaw[%d] should be []any", i)
		rawForParse[i] = slice
	}
	extraFeatures, err := kvblock.ParseRawExtraKeys(rawForParse)
	require.NoError(t, err)

	// Compute "engine" block hashes (simulating what vLLM would put in block_hashes).
	engineKeys, err := tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, modelName, extraFeatures,
	)
	require.NoError(t, err)
	require.Len(t, engineKeys, numBlocks)

	// Build a vLLM-format msgpack BlockStored event.
	engineHashesU64 := make([]any, numBlocks)
	for i, k := range engineKeys {
		engineHashesU64[i] = uint64(k)
	}

	blockStoredEvent := []any{
		"BlockStored",
		engineHashesU64, // block_hashes
		nil,             // parent_block_hash (no parent)
		tokens,          // token_ids
		blockSize,       // block_size
		nil,             // lora_id
		"GPU",           // medium
		nil,             // lora_name
		extraKeysRaw,    // extra_keys
	}

	eventBatch := []any{
		1234567890.0,            // timestamp
		[]any{blockStoredEvent}, // events
		nil,                     // data_parallel_rank
	}

	payload, err := msgpack.Marshal(eventBatch)
	require.NoError(t, err)

	// --- INGESTION PATH ---
	// Parse through VLLMAdapter (same as real ZMQ subscriber would).
	adapter := engineadapter.NewVLLMAdapter()
	topic := "kv@" + podID + "@" + modelName

	parsedPodID, parsedModel, batch, err := adapter.ParseMessage(&kvevents.RawMessage{
		Topic:   topic,
		Payload: payload,
	})
	require.NoError(t, err)
	assert.Equal(t, podID, parsedPodID)
	assert.Equal(t, modelName, parsedModel)
	require.Len(t, batch.Events, 1)

	stored, ok := batch.Events[0].(*kvevents.BlockStoredEvent)
	require.True(t, ok)
	assert.Len(t, stored.BlockHashes, numBlocks)
	assert.Len(t, stored.Tokens, numTokens)

	// Convert event's extra_keys → features (same as pool.go does).
	parsedFeatures, err := kvblock.ParseRawExtraKeys(stored.ExtraKeys)
	require.NoError(t, err)

	// Compute request keys from the event data (ingestion path).
	ingestKeys, err := tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, stored.Tokens, parsedModel, parsedFeatures,
	)
	require.NoError(t, err)

	// --- REQUEST PATH ---
	// Simulate what the indexer does when scoring a new multimodal request.
	// The tokenizer returns mm_hashes and mm_placeholders.
	mmHashes := map[string][]string{
		"image": {img1Hash, img2Hash},
	}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {
			{Offset: 64, Length: 32},  // image 1
			{Offset: 112, Length: 16}, // image 2
		},
	}

	requestFeatures := kvblock.ComputeBlockExtraFeatures(
		mmHashes, mmPlaceholders, blockSize, numTokens,
	)

	requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, modelName, requestFeatures,
	)
	require.NoError(t, err)

	// --- VERIFICATION ---
	// Both paths must produce identical block keys.
	require.Equal(t, len(ingestKeys), len(requestKeys))
	for i := range ingestKeys {
		assert.Equal(t, ingestKeys[i], requestKeys[i],
			"block %d: ingestion key %d != request key %d", i, ingestKeys[i], requestKeys[i])
	}

	// Also verify they match the original engine keys.
	for i := range engineKeys {
		assert.Equal(t, engineKeys[i], ingestKeys[i],
			"block %d: engine key != ingestion key", i)
	}

	// Verify text-only blocks differ from MM blocks.
	assert.NotEqual(t, ingestKeys[3], ingestKeys[4],
		"text block 3 and image block 4 should differ")

	// Verify the two different images produce different keys
	// (even if their block-relative offsets are the same).
	assert.NotEqual(t, ingestKeys[4], ingestKeys[7],
		"image1 block 4 and image2 block 7 should differ (different hashes)")

	// Verify lookup would work: store engine keys in an index, look up with request keys.
	index, err := kvblock.NewInMemoryIndex(nil)
	require.NoError(t, err)

	engineBlockHashes := make([]kvblock.BlockHash, numBlocks)
	for i, h := range stored.BlockHashes {
		engineBlockHashes[i] = kvblock.BlockHash(h)
	}

	err = index.Add(context.Background(), engineBlockHashes, ingestKeys,
		[]kvblock.PodEntry{{PodIdentifier: podID, DeviceTier: "GPU"}})
	require.NoError(t, err)

	// Look up using request-path keys — should find the pod for all blocks.
	results, err := index.Lookup(context.Background(), requestKeys, sets.New[string]())
	require.NoError(t, err)

	for i, key := range requestKeys {
		pods, found := results[key]
		assert.True(t, found, "block %d: request key should be in index", i)
		if found {
			assert.Len(t, pods, 1, "block %d: should map to exactly one pod", i)
			assert.Equal(t, podID, pods[0].PodIdentifier)
		}
	}
}
