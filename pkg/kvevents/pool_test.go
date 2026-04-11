package kvevents

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// newCanonicalTestPool creates a Pool with real InMemoryIndex and
// ChunkedTokenDatabase, configured for canonical block size testing.
func newCanonicalTestPool(t *testing.T, canonicalBlockSize int) (
	*Pool, kvblock.Index, kvblock.TokenProcessor,
) {
	t.Helper()

	idx, err := kvblock.NewInMemoryIndex(kvblock.DefaultInMemoryIndexConfig())
	require.NoError(t, err)

	tp, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
		BlockSize: 16,
		HashSeed:  "test",
	})
	require.NoError(t, err)

	cfg := DefaultConfig()
	cfg.CanonicalBlockSize = canonicalBlockSize

	pool := NewPool(cfg, idx, tp, nil)
	return pool, idx, tp
}

// makeTokens creates a token slice [1, 2, ..., n].
func makeTokens(n int) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = uint32(i + 1)
	}
	return tokens
}

// makeEngineKeys creates engine key slice [base, base+1, ..., base+n-1].
func makeEngineKeys(n int, base uint64) []uint64 {
	keys := make([]uint64, n)
	for i := range keys {
		keys[i] = base + uint64(i)
	}
	return keys
}

// TestCanonicalWritePath_FallbackLegacy verifies that when canonicalBlockSize is 0,
// the pool takes the legacy path: engine keys are passed to Index.Add and the pool's LRU is unused.
func TestCanonicalWritePath_FallbackLegacy(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, _ := newCanonicalTestPool(t, 0) // canonicalBlockSize = 0

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 500) // 4 keys, engine block size 16

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-legacy", "test-model")

	// Verify engine->request mapping exists in the INDEX (not pool LRU)
	// This only works if Add() received non-nil engineKeys (legacy path)
	for _, ek := range engineKeys {
		reqKey, err := idx.GetRequestKey(ctx, kvblock.BlockHash(ek))
		require.NoError(t, err, "engine key %d should be resolvable via index", ek)
		assert.NotEqual(t, kvblock.EmptyBlockHash, reqKey)
	}

	// Verify pool's LRU is empty (canonical mapping not used in legacy path)
	assert.Equal(t, 0, pool.engineToCanonicalKeys.Len(), "pool LRU should be empty in legacy path")
}

// TestCanonicalWritePath_ManyToOne verifies the many:1 mapping when engine block size (16)
// is smaller than canonical (64): 4 engine keys map to 1 canonical request key.
func TestCanonicalWritePath_ManyToOne(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newCanonicalTestPool(t, 64)

	// 128 tokens, 8 engine keys -> engine block size 16
	// canonical block size = 64 -> 2 full canonical keys
	// 4 engine keys per canonical key (many:1)
	tokens := makeTokens(128)
	engineKeys := makeEngineKeys(8, 100)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-a", "test-model")

	// Compute expected canonical keys independently
	canonicalKeys, err := tp.TokensToKVBlockKeysAtBlockSize(
		kvblock.EmptyBlockHash, tokens, "test-model", nil, 64)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Verify both canonical keys are in the index with pod-a
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "canonical key should have exactly one pod")
		assert.Equal(t, "pod-a", result[ck][0].PodIdentifier)
	}

	// Verify all 8 engine keys are in the LRU
	for _, ek := range engineKeys {
		mapped, found := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(ek))
		require.True(t, found, "engine key %d should be in LRU", ek)
		require.Len(t, mapped, 1, "many:1 -> each engine key maps to 1 canonical key")
	}

	// Verify engine keys 0-3 are in canonical[0] and 4-7 in canonical[1]
	for i := 0; i < 4; i++ {
		mapped, _ := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[i]))
		assert.Equal(t, canonicalKeys[0], mapped[0],
			"engine key %d should map to canonical key 0", i)
	}

	for i := 4; i < 8; i++ {
		mapped, _ := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[i]))
		assert.Equal(t, canonicalKeys[1], mapped[0],
			"engine key %d should map to canonical key 1", i)
	}
}

// TestCanonicalWritePath_OneToMany verifies the 1:many mapping when engine block size (128)
// is larger than canonical (64): 1 engine key maps to 2 canonical request keys.
func TestCanonicalWritePath_OneToMany(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newCanonicalTestPool(t, 64)

	// 256 tokens, 2 engine keys -> engine block size 128
	// canonical block size = 64 -> 4 full canonical keys
	// Each engine keys covers two canonical keys (1:many)
	tokens := makeTokens(256)
	engineKeys := makeEngineKeys(2, 200)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-b", "test-model")

	// Compute expected canonical keys independently
	canonicalKeys, err := tp.TokensToKVBlockKeysAtBlockSize(
		kvblock.EmptyBlockHash, tokens, "test-model", nil, 64)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Verify all 4 canonical keys are in the index with pod-b
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "canonical key should have exactly one pod")
		assert.Equal(t, "pod-b", result[ck][0].PodIdentifier)
	}

	// Verify engine key 0 maps to canonical keys [0, 1]
	mapped0, found := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[0]))
	require.True(t, found)
	require.Len(t, mapped0, 2, "1:many -> engine key 0 maps to 2 canonical keys")
	assert.Equal(t, canonicalKeys[0], mapped0[0])
	assert.Equal(t, canonicalKeys[1], mapped0[1])

	// Verify engine key 1 maps to canonical keys [2, 3]
	mapped1, found := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[1]))
	require.True(t, found)
	require.Len(t, mapped1, 2, "1:many -> engine key 0 maps to 2 canonical keys")
	assert.Equal(t, canonicalKeys[2], mapped1[0])
	assert.Equal(t, canonicalKeys[3], mapped1[1])
}

// TestCanonicalEviction_Eager verifies eager eviction: removing one engine key evicts its
// mapped canonical key from the index while leaving unrelated canonical keys intact.
func TestCanonicalEviction_Eager(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newCanonicalTestPool(t, 64)

	// The eager eviction policy: when ANY engine key within a canonical block is evicted,
	// ALL mapped canonical keys must be removed

	// 128 tokens, 8 engine keys -> engine block size 16
	// canonical block size = 64 -> 2 full canonical keys
	// 4 engine keys per canonical key (many:1)
	tokens := makeTokens(128)
	engineKeys := makeEngineKeys(8, 100)

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-a", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeysAtBlockSize(
		kvblock.EmptyBlockHash, tokens, "test-model", nil, 64)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Evict engineKey 0 which maps to canonical key 0
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: []uint64{engineKeys[0]},
			},
		},
	}
	pool.processEventBatch(ctx, removeBatch, "pod-a", "test-model")

	// Verify canonical key 0 is evicted
	result0, err := idx.Lookup(ctx, []kvblock.BlockHash{canonicalKeys[0]}, nil)
	require.NoError(t, err)
	assert.Empty(t, result0[canonicalKeys[0]], "canonical key 0 should be evicted after engine key 0 removal")

	// Verify canonical key 1 still present
	result1, err := idx.Lookup(ctx, []kvblock.BlockHash{canonicalKeys[1]}, nil)
	require.NoError(t, err)
	assert.Len(t, result1[canonicalKeys[1]], 1, "canonical key 1 should still have pod-a")

	// Verify engine key 0 removed from LRU
	_, found := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[0]))
	assert.False(t, found, "evicted engine key should be removed from LRU")

	// Verify other engine keys still in LRU
	_, found = pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[1]))
	assert.True(t, found, "non-evicted engine keys should remain in LRU")
}

// TestCanonicalEviction_OneToMany verifies that evicting a single engine key that maps to
// multiple canonical keys (1:many) removes all of them from the index.
func TestCanonicalEviction_OneToMany(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newCanonicalTestPool(t, 64)

	// 256 tokens, 2 engine keys -> engine block size 128
	// canonical block size = 64 -> 4 full canonical keys
	// Each engine keys covers two canonical keys (1:many)
	tokens := makeTokens(256)
	engineKeys := makeEngineKeys(2, 300)

	storeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, storeBatch, "pod-c", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeysAtBlockSize(kvblock.EmptyBlockHash, tokens, "test-model", nil, 64)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Evict engine key 0 should remove canonical keys 0 AND 1
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: []uint64{engineKeys[0]},
			},
		},
	}
	pool.processEventBatch(ctx, removeBatch, "pod-c", "test-model")

	// Verify canonical keys 0 and 1 evicted
	for _, ck := range canonicalKeys[:2] {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "canonical key mapped to evicted engine key should be gone")
	}

	// Verify canonical keys 2 and 3 still present
	for _, ck := range canonicalKeys[2:] {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Len(t, result[ck], 1, "canonical keys mapped to non-evicted engine key should remain")
	}

	// Verify engine key 0 removed from LRU, engine key 1 remains
	_, found := pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[0]))
	assert.False(t, found)
	_, found = pool.engineToCanonicalKeys.Get(kvblock.BlockHash(engineKeys[1]))
	assert.True(t, found)
}

// TestCanonicalWritePath_CrossEngineScoring verifies that two engines with different block sizes
// (16 and 32) storing the same tokens produce identical canonical keys, so both pods appear in lookups.
func TestCanonicalWritePath_CrossEngineScoring(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newCanonicalTestPool(t, 64)

	tokens := makeTokens(128)

	// Engine A: block size 16, 8 engine keys
	batchA := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: makeEngineKeys(8, 100),
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batchA, "pod-a", "test-model")

	// Engine B: block size 32, 4 engine keys
	batchB := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: makeEngineKeys(4, 200),
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batchB, "pod-b", "test-model")

	// Both produce the same 2 canonical keys
	canonicalKeys, err := tp.TokensToKVBlockKeysAtBlockSize(
		kvblock.EmptyBlockHash, tokens, "test-model", nil, 64)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 2)

	// Both pods should appear under each canonical key
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		pods := result[ck]
		require.Len(t, pods, 2, "both pods should be present")

		podIDs := map[string]bool{}
		for _, p := range pods {
			podIDs[p.PodIdentifier] = true
		}
		assert.True(t, podIDs["pod-a"], "pod-a should be present")
		assert.True(t, podIDs["pod-b"], "pod-b should be present")
	}
}

// TestCanonicalEviction_UnknownEngineKey verifies that evicting an engine key not in the
// pool's LRU is a no-op — no panic, no error.
func TestCanonicalEviction_UnknownEngineKey(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, _, _ := newCanonicalTestPool(t, 64)

	// Evict an engine key that was never stored
	removeBatch := &EventBatch{
		Events: []GenericEvent{
			&BlockRemovedEvent{
				BlockHashes: []uint64{999999},
			},
		},
	}

	// Should not panic or error, just skip
	assert.NotPanics(t, func() {
		pool.processEventBatch(ctx, removeBatch, "pod-x", "test-model")
	})
}

// TestCanonicalWritePath_PartialBlockDrop verifies that tokens fewer than the canonical block
// size produce zero canonical keys and the event is silently skipped.
func TestCanonicalWritePath_PartialBlockDrop(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, _ := newCanonicalTestPool(t, 64)

	// 48 tokens < canonical block size (64), so 0 canonical keys
	tokens := makeTokens(48)
	engineKeys := makeEngineKeys(3, 400) // 3 keys -> engine block size 16

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: engineKeys,
				Tokens:      tokens,
				ParentHash:  0,
			},
		},
	}
	pool.processEventBatch(ctx, batch, "pod-partial", "test-model")

	// Verify nothing was added to the index
	// Lookup any key, should be empty
	result, err := idx.Lookup(ctx, []kvblock.BlockHash{kvblock.BlockHash(1)}, nil)
	require.NoError(t, err)
	assert.Empty(t, result[kvblock.BlockHash(1)])

	// Verify pool LRU is empty (no mappings created)
	assert.Equal(t, 0, pool.engineToCanonicalKeys.Len())
}
