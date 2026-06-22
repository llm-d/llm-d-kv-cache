package kvevents //nolint:testpackage // tests use unexported eventDedupFilter and processEventBatch

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

func testScope(pod, tier string, group, dpRank int) blockScope {
	return blockScope{
		podIdentifier:    pod,
		deviceTier:       tier,
		groupIdx:         group,
		dataParallelRank: dpRank,
	}
}

func gpuScope(pod string) blockScope {
	return testScope(pod, "gpu", noGroupIdx, noDataParallelRank)
}

// TestEventDedupFilter_DuplicateStoreSuppressesFirstRemove is the core contract:
// two announcements of the same hashes (overlapping offloaded chunks) require
// two removes before the index eviction is forwarded.
func TestEventDedupFilter_DuplicateStoreSuppressesFirstRemove(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")
	hashes := []uint64{1, 2, 3}

	f.trackStore(s, hashes)
	f.trackStore(s, hashes) // sibling chunk re-announces the same constituent hashes

	assert.Empty(t, f.filterRemove(s, hashes),
		"first of two duplicate removes must be fully suppressed")
	assert.Equal(t, hashes, f.filterRemove(s, hashes),
		"second remove must forward every hash to the index")
}

// TestEventDedupFilter_AggregatesAcrossSources documents the pod-level intent:
// because the index identity is rank-agnostic on current main, stores that
// share a scope (e.g. different data-parallel ranks, both using the sentinel)
// aggregate into one count, so the block is only evicted once every reference
// is released.
func TestEventDedupFilter_AggregatesAcrossSources(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	f.trackStore(s, []uint64{7}) // source 1
	f.trackStore(s, []uint64{7}) // source 2, same scope

	assert.Empty(t, f.filterRemove(s, []uint64{7}), "one reference still outstanding")
	assert.Equal(t, []uint64{7}, f.filterRemove(s, []uint64{7}), "last reference released")
}

// TestEventDedupFilter_TierIndependence verifies a CPU store cannot mask a GPU
// remove (and vice versa): the index keeps gpu and cpu copies of a hash as
// distinct entries, so their reference counts must be independent.
func TestEventDedupFilter_TierIndependence(t *testing.T) {
	f := newEventDedupFilter()
	gpu := testScope("pod-a", "gpu", noGroupIdx, noDataParallelRank)
	cpu := testScope("pod-a", "cpu", noGroupIdx, noDataParallelRank)

	f.trackStore(gpu, []uint64{1})
	f.trackStore(cpu, []uint64{1})

	// A single cpu remove must forward (cpu count 1->0); if the counts were
	// shared this would be suppressed (2->1) and kept empty.
	assert.Equal(t, []uint64{1}, f.filterRemove(cpu, []uint64{1}),
		"cpu remove must not be masked by the gpu store")
	// The gpu reference is still outstanding and forwards independently.
	assert.Equal(t, []uint64{1}, f.filterRemove(gpu, []uint64{1}),
		"gpu remove must still forward after the cpu remove")
}

// TestEventDedupFilter_GroupIndependence verifies different KV-cache groups are
// reference-counted independently, mirroring distinct grouped PodEntries.
func TestEventDedupFilter_GroupIndependence(t *testing.T) {
	f := newEventDedupFilter()
	g0 := testScope("pod-a", "gpu", 0, noDataParallelRank)
	g1 := testScope("pod-a", "gpu", 1, noDataParallelRank)

	f.trackStore(g0, []uint64{1})
	f.trackStore(g1, []uint64{1})

	assert.Equal(t, []uint64{1}, f.filterRemove(g1, []uint64{1}),
		"group 1 remove must be independent of the group 0 store")
}

// TestEventDedupFilter_DataParallelRankIndependence verifies the filter already
// separates reference counts by data-parallel rank, so it is ready to become
// DP-aware (see noDataParallelRank / PR #370) once the pool feeds a real rank
// instead of the sentinel. On current main every scope uses the sentinel, so
// this dimension is dormant in production but unit-tested here for forward
// compatibility.
func TestEventDedupFilter_DataParallelRankIndependence(t *testing.T) {
	f := newEventDedupFilter()
	dp0 := testScope("pod-a", "gpu", noGroupIdx, 0)
	dp1 := testScope("pod-a", "gpu", noGroupIdx, 1)

	f.trackStore(dp0, []uint64{1})
	f.trackStore(dp1, []uint64{1})

	assert.Equal(t, []uint64{1}, f.filterRemove(dp1, []uint64{1}),
		"rank 1 remove must be independent of the rank 0 store")
}

// TestEventDedupFilter_UnknownRemovePassesThrough verifies defensive
// pass-through for never-seen hashes and that the count never goes negative.
func TestEventDedupFilter_UnknownRemovePassesThrough(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	assert.Equal(t, []uint64{42}, f.filterRemove(s, []uint64{42}),
		"unknown remove must pass through")
	assert.Equal(t, []uint64{42}, f.filterRemove(s, []uint64{42}),
		"repeated unknown remove must keep passing through (no negative count)")

	// A store after underflow attempts still yields a clean single reference.
	f.trackStore(s, []uint64{42})
	assert.Equal(t, []uint64{42}, f.filterRemove(s, []uint64{42}),
		"store must establish a fresh single reference")
}

// TestEventDedupFilter_PartialForward verifies a mixed remove forwards only the
// hashes that are released or unknown, preserving input order.
func TestEventDedupFilter_PartialForward(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	f.trackStore(s, []uint64{1}) // hash 1: count 2 (still referenced after one remove)
	f.trackStore(s, []uint64{1})
	f.trackStore(s, []uint64{2}) // hash 2: count 1 (released by one remove)

	// hash 1 suppressed (2->1), hash 2 forwarded (1->0), hash 3 unknown forwarded.
	assert.Equal(t, []uint64{2, 3}, f.filterRemove(s, []uint64{1, 2, 3}))
}

// TestEventDedupFilter_ClearResets verifies clear zeroes a pod's counts.
func TestEventDedupFilter_ClearResets(t *testing.T) {
	f := newEventDedupFilter()
	s := gpuScope("pod-a")

	f.trackStore(s, []uint64{1})
	f.trackStore(s, []uint64{1}) // count 2

	f.clear("pod-a")

	// If clear had not reset the count, this remove would be suppressed (2->1).
	assert.Equal(t, []uint64{1}, f.filterRemove(s, []uint64{1}),
		"clear must reset the count so the remove passes through")
}

// TestEventDedupFilter_ClearIsolatesPods verifies clearing one pod does not
// affect another pod's counts.
func TestEventDedupFilter_ClearIsolatesPods(t *testing.T) {
	f := newEventDedupFilter()
	a := gpuScope("pod-a")
	b := gpuScope("pod-b")

	f.trackStore(a, []uint64{1})
	f.trackStore(a, []uint64{1})
	f.trackStore(b, []uint64{1})
	f.trackStore(b, []uint64{1})

	f.clear("pod-a")

	assert.Equal(t, []uint64{1}, f.filterRemove(a, []uint64{1}),
		"cleared pod-a count must be reset")
	assert.Empty(t, f.filterRemove(b, []uint64{1}),
		"pod-b count must survive pod-a clear (first of two removes suppressed)")
}

// TestEventDedupFilter_NilSafe verifies the nil-receiver guards so a Pool
// without a filter degrades to forwarding every event.
func TestEventDedupFilter_NilSafe(t *testing.T) {
	var f *eventDedupFilter
	s := gpuScope("pod-a")

	assert.NotPanics(t, func() { f.trackStore(s, []uint64{1}) })
	assert.NotPanics(t, func() { f.clear("pod-a") })
	assert.Equal(t, []uint64{1}, f.filterRemove(s, []uint64{1}),
		"nil filter must forward removes unchanged")
}

func TestGroupIdxOrNoGroup(t *testing.T) {
	assert.Equal(t, noGroupIdx, groupIdxOrNoGroup(nil))
	g := 3
	assert.Equal(t, 3, groupIdxOrNoGroup(&g))
}

// TestPool_DuplicateStoreSurvivesFirstRemove is the end-to-end proof through
// processEventBatch with a real index: two overlapping chunks announce the same
// blocks, so the first BlockRemoved must not evict them and the second must.
func TestPool_DuplicateStoreSurvivesFirstRemove(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 800)

	store := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
			},
		}, "pod-dup", "test-model")
	}
	remove := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{BlockHashes: engineKeys},
			},
		}, "pod-dup", "test-model")
	}

	store() // overlapping chunk A announces these constituent hashes
	store() // overlapping chunk B re-announces the same hashes

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// First remove (chunk A evicted) must NOT drop the blocks.
	remove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "block must survive the first of two duplicate removes")
	}

	// Second remove (chunk B evicted) drops them.
	remove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "block must be evicted after the second duplicate remove")
	}

	// Engine->request mapping should also be gone after full eviction.
	_, err = idx.GetRequestKey(ctx, kvblock.BlockHash(engineKeys[0]))
	assert.Error(t, err, "engine->request mapping should be removed after the second remove")
}

// TestPool_DuplicateCPUOffloadRemovalSurvivesFirstRemove exercises the
// offloading wiring end to end through processEventBatch: a CPU tier is
// established via the empty-token device-tier update path, re-announced by two
// overlapping chunks, and must require two CPU removes before the CPU entry is
// evicted — while the independently reference-counted GPU entry (never removed)
// is untouched throughout. This jointly exercises handleDeviceTierUpdate, the
// deviceTier normalization on both store and remove, and the dedup scope.
func TestPool_DuplicateCPUOffloadRemovalSurvivesFirstRemove(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 900)

	// Step 1: GPU store with tokens establishes the engine->request mapping
	// that the empty-token CPU offload path resolves against.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
		},
	}, "pod-offload", "test-model")

	// Step 2: the same CPU offload (empty tokens, "CPU" tier) announced twice,
	// as two overlapping chunks would re-announce the shared constituent hashes.
	cpuStore := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{BlockHashes: engineKeys, Tokens: nil, ParentHash: 0, DeviceTier: "CPU"},
			},
		}, "pod-offload", "test-model")
	}
	cpuStore()
	cpuStore()

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)
	require.Len(t, canonicalKeys, 4)

	// Both gpu and cpu entries should now exist for each canonical key.
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 2, "gpu and cpu entries should both exist after offload")
	}

	cpuRemove := func() {
		pool.processEventBatch(ctx, &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{BlockHashes: engineKeys, DeviceTier: "CPU"},
			},
		}, "pod-offload", "test-model")
	}

	// Step 3: the first CPU remove (one overlapping chunk evicted) must be
	// suppressed — the cpu entry still has an outstanding reference.
	cpuRemove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 2, "cpu entry must survive the first of two duplicate CPU removes")
		tiers := map[string]bool{}
		for _, pe := range result[ck] {
			tiers[pe.DeviceTier] = true
		}
		assert.True(t, tiers["cpu"], "cpu entry should still be present after the first CPU remove")
		assert.True(t, tiers["gpu"], "gpu entry should be untouched")
	}

	// Step 4: the second CPU remove releases the last reference and evicts the
	// cpu entry; the gpu entry (never removed) must remain.
	cpuRemove()
	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		require.Len(t, result[ck], 1, "only the gpu entry should remain after the second CPU remove")
		assert.Equal(t, "gpu", result[ck][0].DeviceTier, "surviving entry must be the untouched gpu copy")
	}
}

// TestPool_AllBlocksClearedResetsDedup verifies the filter is reset on
// AllBlocksCleared, so a post-clear store/remove cycle behaves freshly rather
// than carrying a stale reference that would suppress the remove.
func TestPool_AllBlocksClearedResetsDedup(t *testing.T) {
	ctx := logging.NewTestLoggerIntoContext(context.Background())
	pool, idx, tp := newTestPool(t, 16)

	tokens := makeTokens(64)
	engineKeys := makeEngineKeys(4, 850)

	storeTwice := func() {
		for range 2 {
			pool.processEventBatch(ctx, &EventBatch{
				Events: []GenericEvent{
					&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
				},
			}, "pod-clr", "test-model")
		}
	}

	// Two stores -> reference count 2 -> would normally need two removes.
	storeTwice()

	// Clear wipes both the index and the dedup counts for the pod.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{&AllBlocksClearedEvent{}},
	}, "pod-clr", "test-model")

	// Re-establish a single reference after the clear.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{BlockHashes: engineKeys, Tokens: tokens, ParentHash: 0},
		},
	}, "pod-clr", "test-model")

	canonicalKeys, err := tp.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "test-model", nil)
	require.NoError(t, err)

	// A single remove must now fully evict: if the pre-clear count of 2 had
	// survived, this remove would be suppressed and the blocks would linger.
	pool.processEventBatch(ctx, &EventBatch{
		Events: []GenericEvent{&BlockRemovedEvent{BlockHashes: engineKeys}},
	}, "pod-clr", "test-model")

	for _, ck := range canonicalKeys {
		result, err := idx.Lookup(ctx, []kvblock.BlockHash{ck}, nil)
		require.NoError(t, err)
		assert.Empty(t, result[ck], "single remove after clear must evict (dedup count was reset)")
	}
}
