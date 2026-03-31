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

package kvevents //nolint:testpackage // Tests internal pool event processing

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// TestPoolEventProcessing_GroupIDHandling verifies that:
// 1. Event with group_id AND model is HMA → StoredGroups = []int{group_id}.
// 2. Event with group_id BUT model is non-HMA → StoredGroups = nil (ignore group_id).
// 3. Event without group_id (group_id=0 default) → depends on model config.
func TestPoolEventProcessing_GroupIDHandling(t *testing.T) {
	ctx := context.Background()

	t.Run("HMAModel_WithGroupID_TracksGroup", func(t *testing.T) {
		// Setup: HMA model registry.
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with GroupIdx = 1 for HMA model
		// Need at least 16 tokens to create one complete block (defaultBlockSize = 16)
		// 16 tokens = 1 block, so we need 1 engineKey (BlockHash)
		tokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{100}, // 1 block = 1 hash
					Tokens:      tokens,
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    1, // Group 1
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "DeepSeek-V3")

		// Verify StoredGroups was set to [1]
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "DeepSeek-V3", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		// Should have entries with StoredGroups = [1]
		found := false
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.NotNil(t, entry.StoredGroups, "HMA model MUST have non-nil StoredGroups")
					assert.Equal(t, []int{1}, entry.StoredGroups, "Should track group ID 1")
					found = true
				}
			}
		}
		assert.True(t, found, "Should have found entry with group tracking")
	})

	t.Run("NonHMAModel_WithGroupID_IgnoresGroup", func(t *testing.T) {
		// Setup: Simple (non-HMA) model registry
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "Qwen/Qwen3-8B",
				IsHMA: false,
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with GroupIdx = 1, but model is non-HMA
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{200},
					Tokens:      []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    1, // Should be ignored for non-HMA
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "Qwen/Qwen3-8B")

		// Verify StoredGroups is nil (group_id ignored)
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
			kvblock.EmptyBlockHash,
			[]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			"Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		// Should have entries with StoredGroups = nil
		found := false
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.Nil(t, entry.StoredGroups, "Non-HMA model MUST have nil StoredGroups (ignore group_id)")
					found = true
				}
			}
		}
		assert.True(t, found, "Should have found entry without group tracking")
	})

	t.Run("HMAModel_WithGroupIDZero_TracksGroup", func(t *testing.T) {
		// Setup: HMA model
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
				},
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with GroupIdx = 0 (default) for HMA model
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{300},
					Tokens:      []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    0, // Group 0
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "DeepSeek-V3")

		// Verify StoredGroups = [0]
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
			kvblock.EmptyBlockHash,
			[]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			"DeepSeek-V3", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		found := false
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.NotNil(t, entry.StoredGroups, "HMA model should track groups")
					assert.Equal(t, []int{0}, entry.StoredGroups, "Should track group ID 0")
					found = true
				}
			}
		}
		assert.True(t, found, "Should have found entry with group 0")
	})

	t.Run("NonHMAModel_WithGroupIDZero_IgnoresGroup", func(t *testing.T) {
		// Setup: Non-HMA model
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "Qwen/Qwen3-8B",
				IsHMA: false,
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with GroupIdx = 0 (default)
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{400},
					Tokens:      []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    0,
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "Qwen/Qwen3-8B")

		// Verify StoredGroups is nil
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
			kvblock.EmptyBlockHash,
			[]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			"Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		found := false
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.Nil(t, entry.StoredGroups, "Non-HMA model should have nil StoredGroups")
					found = true
				}
			}
		}
		assert.True(t, found, "Should have found entry")
	})

	t.Run("UnknownModel_WithGroupID_DefaultsToNonHMA", func(t *testing.T) {
		// Setup: Empty registry (all models default to non-HMA)
		modelRegistry := kvcache.NewDefaultModelRegistry()

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with GroupIdx = 1 for unknown model
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{500},
					Tokens:      []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    1, // Should be ignored (unknown model = non-HMA)
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "unknown-model")

		// Verify StoredGroups is nil (unknown model defaults to non-HMA)
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
			kvblock.EmptyBlockHash,
			[]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			"unknown-model", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		found := false
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.Nil(t, entry.StoredGroups, "Unknown model should default to non-HMA (nil StoredGroups)")
					found = true
				}
			}
		}
		assert.True(t, found, "Should have found entry")
	})
}

func TestPoolEventProcessing_MultipleBlockHashes(t *testing.T) {
	ctx := context.Background()

	t.Run("HMAModel_WithMultipleBlocks_TracksAllBlocks", func(t *testing.T) {
		// Setup: HMA model registry
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with multiple blocks (32 tokens = 2 blocks, each 16 tokens)
		tokens := []uint32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, // Block 1
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, // Block 2
		}
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{100, 101}, // 2 blocks
					Tokens:      tokens,
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    1,
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "DeepSeek-V3")

		// Verify both blocks are tracked
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "DeepSeek-V3", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		// Should have 2 entries, one for each block
		assert.Len(t, result, 2, "Should have entries for both blocks")

		// Verify each entry has StoredGroups = [1]
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.NotNil(t, entry.StoredGroups, "HMA model MUST have non-nil StoredGroups")
					assert.Equal(t, []int{1}, entry.StoredGroups, "Should track group ID 1")
				}
			}
		}
	})

	t.Run("NonHMAModel_WithMultipleBlocks_TracksAllBlocks", func(t *testing.T) {
		// Setup: Simple model registry
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "Qwen/Qwen3-8B",
				IsHMA: false,
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Event with multiple blocks (32 tokens = 2 blocks)
		tokens := []uint32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
		}
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{200, 201}, // 2 blocks
					Tokens:      tokens,
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    0,
				},
			},
		}

		pool.processEventBatch(ctx, batch, "test-pod", "Qwen/Qwen3-8B")

		// Verify both blocks are tracked with nil StoredGroups
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)

		assert.Len(t, result, 2, "Should have entries for both blocks")

		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.Nil(t, entry.StoredGroups, "Non-HMA model MUST have nil StoredGroups")
				}
			}
		}
	})
}

func TestPoolEventProcessing_WithParentHash(t *testing.T) {
	ctx := context.Background()

	t.Run("HMAModel_WithParentHash_CreatesChain", func(t *testing.T) {
		// Setup: HMA model registry
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
				},
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// First event: parent block (16 tokens)
		parentTokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		parentBatch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{1000}, // Parent block hash
					Tokens:      parentTokens,
					ParentHash:  0, // Root block
					DeviceTier:  "gpu",
					GroupIdx:    0,
				},
			},
		}

		pool.processEventBatch(ctx, parentBatch, "test-pod", "DeepSeek-V3")

		// Second event: child block with parent hash
		childTokens := []uint32{17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
		childBatch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{1001}, // Child block hash
					Tokens:      childTokens,
					ParentHash:  1000, // References parent block
					DeviceTier:  "gpu",
					GroupIdx:    0,
				},
			},
		}

		pool.processEventBatch(ctx, childBatch, "test-pod", "DeepSeek-V3")

		// Verify both parent and child blocks are stored
		parentRequestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, parentTokens, "DeepSeek-V3", nil)
		require.NoError(t, err)
		parentResult, err := index.Lookup(ctx, parentRequestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, parentResult, 1, "Parent block should be stored")

		// Verify child block is stored with parent reference
		childRequestKeys, err := tokenProcessor.TokensToKVBlockKeys(parentRequestKeys[0], childTokens, "DeepSeek-V3", nil)
		require.NoError(t, err)
		childResult, err := index.Lookup(ctx, childRequestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, childResult, 1, "Child block should be stored")

		// Verify group tracking
		for _, entries := range childResult {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.NotNil(t, entry.StoredGroups)
					assert.Equal(t, []int{0}, entry.StoredGroups)
				}
			}
		}
	})

	t.Run("NonHMAModel_WithParentHash_CreatesChain", func(t *testing.T) {
		// Setup: Simple model registry
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "Qwen/Qwen3-8B",
				IsHMA: false,
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// First event: parent block
		parentTokens := []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}
		parentBatch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{2000},
					Tokens:      parentTokens,
					ParentHash:  0,
					DeviceTier:  "gpu",
					GroupIdx:    0,
				},
			},
		}

		pool.processEventBatch(ctx, parentBatch, "test-pod", "Qwen/Qwen3-8B")

		// Second event: child block with parent hash
		childTokens := []uint32{17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}
		childBatch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{2001},
					Tokens:      childTokens,
					ParentHash:  2000,
					DeviceTier:  "gpu",
					GroupIdx:    0,
				},
			},
		}

		pool.processEventBatch(ctx, childBatch, "test-pod", "Qwen/Qwen3-8B")

		// Verify both blocks stored with nil StoredGroups
		parentRequestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, parentTokens, "Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		parentResult, err := index.Lookup(ctx, parentRequestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, parentResult, 1)

		childRequestKeys, err := tokenProcessor.TokensToKVBlockKeys(parentRequestKeys[0], childTokens, "Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		childResult, err := index.Lookup(ctx, childRequestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, childResult, 1)

		// Verify nil StoredGroups for simple model
		for _, entries := range childResult {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.Nil(t, entry.StoredGroups, "Non-HMA model should have nil StoredGroups")
				}
			}
		}
	})
}

func TestPoolEventProcessing_BlockRemovedEvent(t *testing.T) {
	ctx := context.Background()

	t.Run("HMAModel_RemovesSpecificGroup", func(t *testing.T) {
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Add with group 0
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{600},
					Tokens:      []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, batch, "test-pod", "DeepSeek-V3")

		// Add group 1
		batchEvent := batch.Events[0].(*BlockStoredEvent) //nolint:errcheck // Test - type assertion is safe
		batchEvent.GroupIdx = 1
		pool.processEventBatch(ctx, batch, "test-pod", "DeepSeek-V3")

		// Verify both groups present
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
			kvblock.EmptyBlockHash,
			[]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			"DeepSeek-V3", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.ElementsMatch(t, []int{0, 1}, entry.StoredGroups, "Should have both groups")
				}
			}
		}

		// Remove group 0 only
		removeEvent := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{600},
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, removeEvent, "test-pod", "DeepSeek-V3")

		// Verify only group 1 remains
		result, _ = index.Lookup(ctx, requestKeys, sets.Set[string]{}) //nolint:errcheck // Test cleanup - errors not critical
		for _, entries := range result {
			for _, entry := range entries {
				if entry.PodIdentifier == "test-pod" {
					assert.Equal(t, []int{1}, entry.StoredGroups, "Only group 1 should remain")
				}
			}
		}
	})

	t.Run("NonHMAModel_RemovesEntireEntry", func(t *testing.T) {
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{Name: "Qwen/Qwen3-8B", IsHMA: false},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Add entry
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{700},
					Tokens:      []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, batch, "test-pod", "Qwen/Qwen3-8B")

		// Verify entry exists
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(
			kvblock.EmptyBlockHash,
			[]uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			"Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.NotEmpty(t, result, "Entry should exist")

		// Remove entry (group_id ignored for non-HMA)
		removeEvent := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{700},
					GroupIdx:    0, // Ignored
				},
			},
		}
		pool.processEventBatch(ctx, removeEvent, "test-pod", "Qwen/Qwen3-8B")

		// Verify entry is completely removed
		result, _ = index.Lookup(ctx, requestKeys, sets.Set[string]{}) //nolint:errcheck // Test cleanup - errors not critical
		assert.Empty(t, result, "Entry should be completely removed for non-HMA model")
	})

	t.Run("HMAModel_RemovesMultipleBlocks", func(t *testing.T) {
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{
				Name:  "DeepSeek-V3",
				IsHMA: true,
				AttentionGroups: []kvcache.AttentionGroupConfig{
					{GroupID: 0, AttentionType: kvcache.AttentionTypeFull, BlockSize: 64},
					{GroupID: 1, AttentionType: kvcache.AttentionTypeSlidingWindow, BlockSize: 64, SlidingWindowSize: 4096},
				},
			},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Add multiple blocks (32 tokens = 2 blocks)
		tokens := []uint32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
		}
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{800, 801},
					Tokens:      tokens,
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, batch, "test-pod", "DeepSeek-V3")

		// Verify both blocks are stored
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "DeepSeek-V3", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, result, 2, "Both blocks should be stored")

		// Remove both blocks at once
		removeEvent := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{800, 801}, // Remove multiple blocks
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, removeEvent, "test-pod", "DeepSeek-V3")

		// Verify both blocks are removed
		result, _ = index.Lookup(ctx, requestKeys, sets.Set[string]{}) //nolint:errcheck // Test cleanup - errors not critical
		assert.Empty(t, result, "All blocks should be removed")
	})

	t.Run("NonHMAModel_RemovesMultipleBlocks", func(t *testing.T) {
		modelRegistry := kvcache.NewModelRegistry([]*kvcache.ModelConfig{
			{Name: "Qwen/Qwen3-8B", IsHMA: false},
		})

		index, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
		require.NoError(t, err)
		tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
		require.NoError(t, err)
		cfg := DefaultConfig()
		cfg.ModelRegistry = modelRegistry
		pool := NewPool(cfg, index, tokenProcessor, nil)

		// Add multiple blocks
		tokens := []uint32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
			17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
		}
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{900, 901},
					Tokens:      tokens,
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, batch, "test-pod", "Qwen/Qwen3-8B")

		// Verify blocks are stored
		requestKeys, err := tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "Qwen/Qwen3-8B", nil)
		require.NoError(t, err)
		result, err := index.Lookup(ctx, requestKeys, sets.Set[string]{})
		require.NoError(t, err)
		assert.Len(t, result, 2)

		// Remove multiple blocks
		removeEvent := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{900, 901},
					GroupIdx:    0,
				},
			},
		}
		pool.processEventBatch(ctx, removeEvent, "test-pod", "Qwen/Qwen3-8B")

		// Verify all removed
		result, _ = index.Lookup(ctx, requestKeys, sets.Set[string]{}) //nolint:errcheck // Test cleanup - errors not critical
		assert.Empty(t, result, "All blocks should be removed for non-HMA model")
	})
}
