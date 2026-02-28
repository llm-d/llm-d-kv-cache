//go:build integration

/*
Integration test for DP-aware KV cache scoring pipeline.

Tests the full flow:
  vLLM (DP=2) -> ZMQ events -> VLLMAdapter.ParseMessage -> PodEntry with DataParallelRank
  -> podScoringKey produces @dpN keys -> stripDPRankFromScores collapses to pod-level

Run with:
  go test -tags=integration -v -run TestDP ./pkg/kvcache/
*/

package kvcache

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
)

// TestDPScoringKeyPipeline tests the full pipeline from PodEntry through scoring to
// scheduler-side score collapsing for Internal LB mode.
func TestDPScoringKeyPipeline(t *testing.T) {
	// 1. Create PodEntries as the kv-cache indexer would after receiving DP-annotated events
	podEntries := []kvblock.PodEntry{
		{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu", DataParallelRank: 0},
		{PodIdentifier: "10.0.0.1:8000", DeviceTier: "gpu", DataParallelRank: 1},
		{PodIdentifier: "10.0.0.2:8000", DeviceTier: "gpu", DataParallelRank: kvblock.NoDataParallelRank},
	}

	// 2. Verify String() serialization format
	assert.Equal(t, "10.0.0.1:8000@gpu@dp0", podEntries[0].String())
	assert.Equal(t, "10.0.0.1:8000@gpu@dp1", podEntries[1].String())
	assert.Equal(t, "10.0.0.2:8000@gpu", podEntries[2].String())

	// 3. Verify ParsePodEntry roundtrip
	for _, entry := range podEntries {
		parsed, err := kvblock.ParsePodEntry(entry.String())
		require.NoError(t, err)
		assert.Equal(t, entry.PodIdentifier, parsed.PodIdentifier)
		assert.Equal(t, entry.DeviceTier, parsed.DeviceTier)
		assert.Equal(t, entry.DataParallelRank, parsed.DataParallelRank)
	}

	// 4. Verify podScoringKey produces correct keys (package-private, called directly)
	assert.Equal(t, "10.0.0.1:8000@dp0", podScoringKey(podEntries[0]))
	assert.Equal(t, "10.0.0.1:8000@dp1", podScoringKey(podEntries[1]))
	assert.Equal(t, "10.0.0.2:8000", podScoringKey(podEntries[2]))

	// 5. Simulate raw scores as the scorer would produce
	rawScores := map[string]float64{
		"10.0.0.1:8000@dp0": 3.0, // rank 0 has best cache match
		"10.0.0.1:8000@dp1": 2.0, // rank 1 has some cache match
		"10.0.0.2:8000":     1.5, // non-DP pod (External LB mode)
	}

	// 6. Apply scheduler-side stripDPRankFromScores
	stripped := testStripDPRankFromScores(rawScores)
	t.Logf("Raw scores:     %v", rawScores)
	t.Logf("Stripped scores: %v", stripped)

	// Internal LB: both rank scores collapse to pod-level, highest score wins
	assert.Equal(t, 3.0, stripped["10.0.0.1:8000"],
		"Should keep highest score (rank 0's 3.0) for Internal LB pod")
	// Non-DP pod passes through unchanged
	assert.Equal(t, 1.5, stripped["10.0.0.2:8000"],
		"Non-DP pod score should pass through unchanged")
	// Should only have 2 entries (collapsed from 3)
	assert.Len(t, stripped, 2, "Should collapse 3 entries to 2")
	// No @dp keys remain
	for key := range stripped {
		assert.NotContains(t, key, "@dp", "Stripped scores should not contain @dp suffix")
	}
}

// TestDPMsgpackEventParsing tests that the VLLMAdapter correctly parses
// EventBatch payloads with data_parallel_rank from vLLM DP engines.
func TestDPMsgpackEventParsing(t *testing.T) {
	adapter := engineadapter.NewVLLMAdapter()

	// Build a BlockStored event in vLLM msgpack array format:
	// [tag, block_hashes, parent_hash, token_ids, block_size, lora_id, medium, lora_name, extra_keys]
	blockStoredEvent := []any{
		"BlockStored",
		[]any{uint64(12345678)},
		nil,
		[]any{1, 2, 3, 4},
		16,
		nil,     // lora_id
		"GPU",   // medium
		nil,     // lora_name
		[]any{}, // extra_keys
	}

	for _, tc := range []struct {
		name     string
		dpRank   *int
		wantRank *int
	}{
		{"dp_rank=0", intPtr(0), intPtr(0)},
		{"dp_rank=1", intPtr(1), intPtr(1)},
		{"dp_rank=7 (higher rank)", intPtr(7), intPtr(7)},
		// vLLM 0.16.0 always sends a 3-element array; nil dp_rank means non-DP.
		{"dp_rank=nil (non-DP)", nil, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var dpField any
			if tc.dpRank != nil {
				dpField = *tc.dpRank
			}
			batch := []any{1234567890.0, []any{blockStoredEvent}, dpField}

			payload, err := msgpack.Marshal(batch)
			require.NoError(t, err)

			msg := &kvevents.RawMessage{
				Topic:   "kv@10.0.0.1:8000@TestModel",
				Payload: payload,
			}

			podID, modelName, eventBatch, dpRank, err := adapter.ParseMessage(msg)
			require.NoError(t, err)
			assert.Equal(t, "10.0.0.1:8000", podID)
			assert.Equal(t, "TestModel", modelName)
			assert.Len(t, eventBatch.Events, 1)

			if tc.wantRank != nil {
				require.NotNil(t, dpRank, "Expected dp_rank=%d but got nil", *tc.wantRank)
				assert.Equal(t, *tc.wantRank, *dpRank)
			} else {
				assert.Nil(t, dpRank, "Expected nil dp_rank for non-DP events")
			}
		})
	}
}

// TestDPHybridLBScoring simulates Hybrid LB mode where multiple nodes each have
// local DP ranks. Scores should collapse per-node.
func TestDPHybridLBScoring(t *testing.T) {
	// Node 1: ranks 0,1 — Node 2: ranks 2,3
	rawScores := map[string]float64{
		"node1:8000@dp0": 4.0,
		"node1:8000@dp1": 2.5,
		"node2:8000@dp2": 3.0,
		"node2:8000@dp3": 1.0,
	}

	stripped := testStripDPRankFromScores(rawScores)

	assert.Len(t, stripped, 2)
	assert.Equal(t, 4.0, stripped["node1:8000"], "Node 1 should keep rank 0's score (highest)")
	assert.Equal(t, 3.0, stripped["node2:8000"], "Node 2 should keep rank 2's score (highest)")
}

// TestDPExternalLBScoring simulates External LB mode where each rank is a separate pod.
// No @dp suffix exists — stripDPRankFromScores is a no-op.
func TestDPExternalLBScoring(t *testing.T) {
	rawScores := map[string]float64{
		"10.0.0.1:8000": 3.0, // rank 0 pod
		"10.0.0.2:8000": 2.0, // rank 1 pod
		"10.0.0.3:8000": 1.5, // rank 2 pod
	}

	stripped := testStripDPRankFromScores(rawScores)

	// Should be identical — no @dp suffixes to strip
	assert.Equal(t, rawScores, stripped, "External LB scores should pass through unchanged")
}

// TestDPMixedModes simulates a mixed environment with both DP and non-DP pods.
func TestDPMixedModes(t *testing.T) {
	rawScores := map[string]float64{
		"dp-pod:8000@dp0":     5.0, // Internal LB pod, rank 0
		"dp-pod:8000@dp1":     3.0, // Internal LB pod, rank 1
		"standalone-pod:8000": 4.0, // Non-DP pod
	}

	stripped := testStripDPRankFromScores(rawScores)

	assert.Len(t, stripped, 2)
	assert.Equal(t, 5.0, stripped["dp-pod:8000"])
	assert.Equal(t, 4.0, stripped["standalone-pod:8000"])
}

// --- Helper functions replicating scheduler logic ---

// testStripDPRankFromScores replicates scheduler's pkg/plugins/scorer/utils.go:stripDPRankFromScores
func testStripDPRankFromScores(scores map[string]float64) map[string]float64 {
	stripped := make(map[string]float64, len(scores))
	for key, score := range scores {
		baseKey := testStripDPRankSuffix(key)
		if existing, ok := stripped[baseKey]; !ok || score > existing {
			stripped[baseKey] = score
		}
	}
	return stripped
}

// testStripDPRankSuffix replicates scheduler's pkg/common/dp.go:StripDPRankSuffix
func testStripDPRankSuffix(key string) string {
	idx := strings.LastIndex(key, "@dp")
	if idx < 0 {
		return key
	}
	suffix := key[idx+3:]
	for _, c := range suffix {
		if c < '0' || c > '9' {
			return key
		}
	}
	if len(suffix) == 0 {
		return key
	}
	return key[:idx]
}

func intPtr(i int) *int {
	return &i
}
