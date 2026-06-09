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
	"slices"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// HybridPrefixCacheScorer scores pods for HMA models using per-group boundary
// evaluation. Full attention acts as a kill switch (must be consecutive from
// block 0), SWA groups track contiguous runs against their window threshold.
// Final score = min(fullLastSeq, min(swaLastSeqs)) + 1.
//
// When attentionInfo is nil it falls back to longest prefix scoring.
type HybridPrefixCacheScorer struct {
	MediumWeights map[string]float64
	DefaultScorer *LongestPrefixScorer
}

func (s *HybridPrefixCacheScorer) Strategy() KVScoringStrategy {
	return HybridPrefixMatch
}

func (s *HybridPrefixCacheScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	attentionInfo *kvblock.AttentionInfo,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	logger := log.FromContext(ctx)

	if attentionInfo == nil {
		logger.V(1).Info("no AttentionInfo, using LongestPrefix scorer")
		return s.DefaultScorer.Score(ctx, keys, keyToPods, nil)
	}

	logger.V(1).Info("using HybridPrefix scorer")
	return s.hybridScore(keys, keyToPods, attentionInfo), nil
}

type podHMAState struct {
	fullActive   bool
	fullLastSeq  int
	swaCount     []int
	swaLastSeq   []int     // -1 = threshold never met
	swaFromStart []bool    // true = SWA group contiguous from block 0 (set false permanently on gap)
	blockWeights []float64 // per-block max device-tier weight for weighted scoring
}

// collectGroups builds a map of podIdentifier → set of GroupIDs present at a block.
func collectGroups(entries []kvblock.PodEntry) map[string][]int {
	groups := make(map[string][]int)
	for _, entry := range entries {
		if entry.HasGroup {
			gid := int(entry.GroupIdx)
			gids := groups[entry.PodIdentifier]
			if !slices.Contains(gids, gid) {
				groups[entry.PodIdentifier] = append(gids, gid)
			}
		}
	}
	return groups
}

func (s *HybridPrefixCacheScorer) hybridScore(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	info *kvblock.AttentionInfo,
) map[string]float64 {
	numSWA := len(info.SWAGroupIDs)
	numBlocks := len(keys)
	states := make(map[string]*podHMAState)

	// Compute per-pod weights for block 0.
	block0Weights := make(map[string]float64)
	fillMaxWeights(block0Weights, keyToPods[keys[0]], s.MediumWeights)

	block0Groups := collectGroups(keyToPods[keys[0]])
	for podID, gids := range block0Groups {
		if !slices.Contains(gids, info.FullGroupID) {
			continue
		}
		st := &podHMAState{
			fullActive:   true,
			fullLastSeq:  0,
			swaCount:     make([]int, numSWA),
			swaLastSeq:   make([]int, numSWA),
			swaFromStart: make([]bool, numSWA),
			blockWeights: make([]float64, numBlocks),
		}
		if w, ok := block0Weights[podID]; ok {
			st.blockWeights[0] = w
		} else {
			st.blockWeights[0] = 1.0
		}
		for i, swaGID := range info.SWAGroupIDs {
			st.swaLastSeq[i] = -1
			if slices.Contains(gids, swaGID) {
				st.swaCount[i] = 1
				st.swaFromStart[i] = true
				if st.swaCount[i] >= info.SWAWindowBlocks[i] {
					st.swaLastSeq[i] = 0
				}
			}
		}
		states[podID] = st
	}

	curWeights := make(map[string]float64)

	for blockIdx := 1; blockIdx < numBlocks; blockIdx++ {
		if len(states) == 0 {
			break
		}

		podGroups := collectGroups(keyToPods[keys[blockIdx]])

		clear(curWeights)
		fillMaxWeights(curWeights, keyToPods[keys[blockIdx]], s.MediumWeights)

		for podID, st := range states {
			gids := podGroups[podID]
			present := len(gids) > 0

			if w, ok := curWeights[podID]; ok {
				st.blockWeights[blockIdx] = w
			}

			if st.fullActive {
				if present && slices.Contains(gids, info.FullGroupID) {
					st.fullLastSeq = blockIdx
				} else {
					st.fullActive = false
				}
			}

			for i, swaGID := range info.SWAGroupIDs {
				if present && slices.Contains(gids, swaGID) {
					st.swaCount[i]++
					if st.swaCount[i] >= info.SWAWindowBlocks[i] {
						st.swaLastSeq[i] = blockIdx
					}
				} else {
					st.swaCount[i] = 0
					st.swaFromStart[i] = false
				}
			}
		}
	}

	scores := make(map[string]float64)
	for podID, st := range states {
		checkpoint := st.fullLastSeq
		drop := false
		for i := range numSWA {
			if st.swaLastSeq[i] < 0 {
				if st.swaFromStart[i] {
					// Prefix shorter than window, but SWA contiguous from
					// block 0 throughout — not a constraint.
					continue
				}
				drop = true
				break
			}
			if st.swaLastSeq[i] < checkpoint {
				checkpoint = st.swaLastSeq[i]
			}
		}
		if drop {
			continue
		}

		var score float64
		for b := 0; b <= checkpoint; b++ {
			w := st.blockWeights[b]
			if w == 0 {
				w = 1.0
			}
			score += w
		}
		scores[podID] = score
	}

	return scores
}
