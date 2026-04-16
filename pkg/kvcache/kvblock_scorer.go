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
	"strconv"
	"strings"

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
	// The returned map is keyed by pod scoring key (not the raw pod identifier):
	// for non-DP pods the key is the pod identifier; for DP-aware pods the key is
	// "<pod>@dp<rank>". Use ParsePodScoringKey to decompose these keys.
	Score(ctx context.Context, keys []kvblock.BlockHash,
		keyToPods map[kvblock.BlockHash][]kvblock.PodEntry) (map[string]float64, error)
}

// NewKVBlockScorer creates a new KVBlockScorer based on the provided strategy.
func NewKVBlockScorer(config *KVBlockScorerConfig) (KVBlockScorer, error) {
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
	// mediumWeights maps medium/device tier names to their scoring weights
	MediumWeights map[string]float64
}

// Strategy returns the strategy type: LongestPrefixMatch.
func (s *LongestPrefixScorer) Strategy() KVScoringStrategy {
	return LongestPrefixMatch
}

// podScoringKey returns a scoring identity for a PodEntry.
// It combines PodIdentifier and DataParallelRank:
//   - "pod-1" when DataParallelRank == NoDataParallelRank (backward compatible)
//   - "pod-1@dp0" when DataParallelRank == 0
func podScoringKey(entry kvblock.PodEntry) string {
	if entry.DataParallelRank == kvblock.NoDataParallelRank {
		return entry.PodIdentifier
	}
	return entry.PodIdentifier + "@dp" + strconv.Itoa(entry.DataParallelRank)
}

// ParsePodScoringKey decomposes a scoring key produced by podScoringKey into
// its pod identifier and optional DP rank. It is the inverse of podScoringKey
// and MUST be the only place that interprets the scoring-key format, so the
// encoding and decoding rules cannot drift.
//
// Behavior:
//   - "pod-1"        -> ("pod-1", nil)          // non-DP
//   - "pod-1@dp0"    -> ("pod-1", *int32 = 0)   // DP rank 0
//   - "pod-1@dp7"    -> ("pod-1", *int32 = 7)   // DP rank 7
//   - "pod@dp"       -> ("pod@dp", nil)         // malformed: no digits
//   - "pod@dpXYZ"    -> ("pod@dpXYZ", nil)      // malformed: non-digit suffix
//   - "pod@dp-1"     -> ("pod@dp-1", nil)       // malformed: negative rank
//
// The caller must treat a nil rank as "non-DP deployment"; a returned zero
// value (*rank == 0) is a valid DP rank, not "absent". Negative ranks are
// treated as malformed (not returned) because the proto contract and
// podScoringKey only ever produce non-negative ranks.
func ParsePodScoringKey(scoringKey string) (pod string, dpRank *int32) {
	idx := strings.LastIndex(scoringKey, "@dp")
	if idx < 0 {
		return scoringKey, nil
	}
	rank, err := strconv.ParseInt(scoringKey[idx+3:], 10, 32)
	if err != nil || rank < 0 {
		// "@dp" not followed by a non-negative integer — treat entire string
		// as pod name so a malformed key cannot leak a negative rank across
		// the gRPC boundary.
		return scoringKey, nil
	}
	r := int32(rank)
	return scoringKey[:idx], &r
}

// fillMaxWeights populates dst with the maximum weight per podID across all
// device tiers for the given entries. The caller must clear dst before calling.
func fillMaxWeights(dst map[string]float64, entries []kvblock.PodEntry, mediumWeights map[string]float64) {
	for _, entry := range entries {
		weight := 1.0
		if mediumWeights != nil {
			if w, exists := mediumWeights[entry.DeviceTier]; exists {
				weight = w
			}
		}
		key := podScoringKey(entry)
		if cur, exists := dst[key]; !exists || weight > cur {
			dst[key] = weight
		}
	}
}

// Score implements the longest prefix scoring logic with weighted sum based on BackendConfig.
// The returned map keys are scoring keys that encode the pod identifier and, when applicable,
// the data parallel rank (e.g., "pod-1" or "pod-1@dp0").
func (s *LongestPrefixScorer) Score(
	_ context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	podScores := make(map[string]float64)

	// Scratch map reused across iterations to avoid per-key allocation.
	curWeights := make(map[string]float64)

	// Build weight index for the first key in a single pass over entries.
	fillMaxWeights(curWeights, keyToPods[keys[0]], s.MediumWeights)

	// activePods tracks pods still in the consecutive prefix chain.
	// Using a plain map and in-place deletion avoids allocating new sets
	// on every iteration.
	activePods := make(map[string]struct{}, len(curWeights))
	for pod, w := range curWeights {
		activePods[pod] = struct{}{}
		podScores[pod] = w
	}

	for i := 1; i < len(keys); i++ {
		if len(activePods) == 0 {
			break
		}

		// Reuse scratch map: clear and refill for current key.
		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[i]], s.MediumWeights)

		// In-place intersection: delete pods from activePods that are not
		// in the current key, and accumulate scores for those that remain.
		for pod := range activePods {
			if w, exists := curWeights[pod]; exists {
				podScores[pod] += w
			} else {
				delete(activePods, pod)
			}
		}
	}

	// Return the map containing the final score for each pod encountered.
	return podScores, nil
}
