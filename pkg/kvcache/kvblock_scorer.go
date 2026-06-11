/*
Copyright 2025 The llm-d Authors.

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

package kvcache

import (
	"context"
	"fmt"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// KVScoringStrategy defines the strategy used to score pods for KV cache block reuse.
type KVScoringStrategy string

const (
	// LongestPrefixMatch Score by longest consecutive match from start.
	LongestPrefixMatch KVScoringStrategy = "LongestPrefix"
)

// KVBlockScorerConfig holds the configuration for the KVBlockScorer.
type KVBlockScorerConfig struct {
	ScoringStrategy KVScoringStrategy
	BackendConfigs  []*KVCacheBackendConfig `json:"backendConfigs"`
}

// DefaultKVBlockScorerConfig returns the default configuration for the KVBlockScorer.
func DefaultKVBlockScorerConfig() *KVBlockScorerConfig {
	return &KVBlockScorerConfig{
		ScoringStrategy: LongestPrefixMatch,
		BackendConfigs:  DefaultKVCacheBackendConfig(),
	}
}

// KVBlockScorer defines the interface for implementing a KV block scoring
// strategy.
type KVBlockScorer interface {
	// Strategy returns the scoring strategy type.
	Strategy() KVScoringStrategy
	// Score scores the blocks based on the scoring strategy.
	// It returns a map of pod names to their scores.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
//
// To enable HMA group-aware scoring, wire the kvevents.Pool's catalog via
// SetGroupCatalog.
func NewKVBlockScorer(config *KVBlockScorerConfig) (*LongestPrefixScorer, error) {
	switch config.ScoringStrategy {
	case LongestPrefixMatch:
		// Build weight map from list of BackendConfigs for efficient lookup
		weightMap := make(map[string]float64)
		for _, medium := range config.BackendConfigs {
			weightMap[medium.Name] = medium.Weight
		}

		return &LongestPrefixScorer{
			MediumWeights: weightMap,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported scoring strategy: %s", config.ScoringStrategy)
	}
}

// LongestPrefixScorer scores based on longest consecutive block matches count
// starting from block 0.
type LongestPrefixScorer struct {
	// MediumWeights maps medium/device tier names to their scoring weights.
	MediumWeights map[string]float64
	// Catalog enables HMA group-aware scoring; may be nil (uniform attention).
	// Wired post-construction from the kvevents.Pool via Indexer.SetGroupCatalog.
	Catalog *kvblock.GroupCatalog
}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

// fillMainWeights populates dst with the maximum weight per pod across the
// given block entries, counting only main-attention entries (full / MLA /
// sink-full). These define the candidate prefix and carry its per-block weight.
// The caller must clear dst before calling so it can be reused across blocks.
//
// vLLM emits one entry per KV cache group for HMA models. Sliding-window entries
// are skipped here — they carry no prefix weight — but they are not ignored by
// scoring overall: Score's phase 2 uses their presence to bound the hit length.
// Mamba, chunked-local, and other kinds are not modeled. Non-HMA entries (no
// group identity) always count, preserving legacy uniform-attention behavior.
// Classification is via kvblock.GroupCatalog.IsMainGroup (nil-safe, with a
// group_idx 0 fallback).
func (s *LongestPrefixScorer) fillMainWeights(dst map[string]float64, entries []kvblock.PodEntry) {
	for _, entry := range entries {
		if entry.HasGroup && !s.Catalog.IsMainGroup(entry.PodIdentifier, entry.GroupIdx) {
			continue
		}
		weight := 1.0
		if s.MediumWeights != nil {
			if w, exists := s.MediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		if cur, exists := dst[entry.PodIdentifier]; !exists || weight > cur {
			dst[entry.PodIdentifier] = weight
		}
	}
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
//
// Scoring mirrors vLLM's hybrid cache-hit convergence in two phases:
//
//  1. Main-attention prefix: the longest run of consecutive blocks from block 0
//     for which the pod holds a main-attention entry (full / MLA / sink-full).
//     Full attention is the binding constraint — it requires the entire prefix.
//
//  2. Sliding-window reduction: for each modeled sliding-window group, a hit
//     needs only a trailing window of contiguous cached blocks, so a right-to-
//     left scan finds the longest prefix whose trailing window is present. This
//     can only shrink the main-attention prefix (the converged min), catching
//     the case where a pod's SWA window for this prefix was evicted while its
//     full-attention blocks survived. Early SWA blocks (outside the window) are
//     not required, matching vLLM. With no catalog (or no modeled sliding-window
//     group) phase 2 is a no-op and the score is the phase-1 prefix.
//
// The score is the tier-weighted sum over the converged hit length.
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	// Phase 1: per-pod main-attention prefix, recording the per-block weight so
	// the prefix can be truncated to the converged hit length in phase 2.
	blockWeights := make(map[string][]float64)

	// Scratch map reused across iterations to avoid per-key allocation.
	curWeights := make(map[string]float64)
	s.fillMainWeights(curWeights, keyToPods[keys[0]])

	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		blockWeights[pod] = []float64{w}
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		clear(curWeights)
		s.fillMainWeights(curWeights, keyToPods[keys[i]])

		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				blockWeights[pod] = append(blockWeights[pod], w)
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Phase 2: reduce each pod's prefix by its sliding-window group coverage,
	// then sum the per-block weights over the converged hit length.
	podScores := make(map[string]float64, len(blockWeights))
	for pod, weights := range blockWeights {
		hit := len(weights)
		// Shrink the hit to the longest prefix whose trailing window is present,
		// per sliding-window group. Exact for the supported case of one modeled
		// SWA group (homogeneous windows). Multiple heterogeneous SWA groups —
		// deferred per issue #336 — would need a fixed-point re-check across
		// groups (vLLM restarts on any shrink); this single pass could otherwise
		// over-count for that case. A nil catalog yields no groups (no-op).
		for _, g := range s.Catalog.SlidingWindowGroups(pod) {
			if l := slidingWindowHitLen(pod, g, hit, keys, keyToPods); l < hit {
				hit = l
			}
		}
		var score float64
		for i := 0; i < hit; i++ {
			score += weights[i]
		}
		podScores[pod] = score
	}

	return podScores, nil
}

// slidingWindowHitLen returns the longest prefix length (in blocks, capped at
// maxBlocks) for which pod has the contiguous trailing window required by the
// sliding-window group g. It mirrors vLLM's SlidingWindowManager.find_longest_cache_hit:
// scan right-to-left and stop once g.ContiguousBlocks blocks are contiguously
// present; if the window never completes, the contiguous run anchored at block 0
// still counts (a prefix shorter than the window).
func slidingWindowHitLen(
	pod string,
	g kvblock.SlidingWindowGroup,
	maxBlocks int,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) int {
	contiguous := 0
	for i := maxBlocks - 1; i >= 0; i-- {
		if groupPresent(keyToPods[keys[i]], pod, g.GroupID) {
			contiguous++
			if contiguous >= g.ContiguousBlocks {
				return i + contiguous
			}
		} else {
			contiguous = 0
		}
	}
	return contiguous
}

// groupPresent reports whether pod holds an entry for KV cache group g at this block.
func groupPresent(entries []kvblock.PodEntry, pod string, g kvblock.GroupID) bool {
	for _, e := range entries {
		if e.PodIdentifier == pod && e.HasGroup && e.GroupIdx == g {
			return true
		}
	}
	return false
}
