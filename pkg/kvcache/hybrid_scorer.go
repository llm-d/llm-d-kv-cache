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

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// HybridPrefixCacheScorer scores pods for HMA models using per-group boundary
// evaluation. Full attention acts as a kill switch (must be consecutive from
// block 0), SWA groups track contiguous runs against their window threshold.
// Final score = min(fullLastSeq, min(swaLastSeqs)) + 1.
//
// When attention info is unavailable for a model it falls back to longest
// prefix scoring.
type HybridPrefixCacheScorer struct {
	MediumWeights map[string]float64
	AttentionInfo map[string]*ModelAttentionInfo
	DefaultScorer *LongestPrefixScorer
}

func (s *HybridPrefixCacheScorer) Strategy() KVScoringStrategy {
	return HybridPrefixMatch
}

func (s *HybridPrefixCacheScorer) Score(
	ctx context.Context,
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	modelName string,
) (map[string]float64, error) {
	if len(keys) == 0 {
		return make(map[string]float64), nil
	}

	logger := log.FromContext(ctx)
	info := s.AttentionInfo[modelName]
	if info == nil {
		logger.V(1).Info("using LongestPrefix scorer", "model", modelName)
		return s.DefaultScorer.Score(ctx, keys, keyToPods, modelName)
	}

	logger.V(1).Info("using HybridPrefix scorer", "model", modelName)
	return s.hybridScore(keys, keyToPods, info), nil
}

type podHMAState struct {
	fullActive  bool
	fullLastSeq int
	swaCount    []int
	swaLastSeq  []int // -1 = threshold never met
}

func (s *HybridPrefixCacheScorer) hybridScore(
	keys []kvblock.BlockHash,
	keyToPods map[kvblock.BlockHash][]kvblock.PodEntry,
	info *ModelAttentionInfo,
) map[string]float64 {
	fullMask := uint32(1) << info.FullGroupID
	numSWA := len(info.SWAGroupIDs)
	swaMasks := make([]uint32, numSWA)
	for i, gid := range info.SWAGroupIDs {
		swaMasks[i] = 1 << gid
	}

	states := make(map[string]*podHMAState)

	for _, entry := range keyToPods[keys[0]] {
		if entry.StoredGroups&fullMask == 0 {
			continue
		}
		st := &podHMAState{
			fullActive:  true,
			fullLastSeq: 0,
			swaCount:    make([]int, numSWA),
			swaLastSeq:  make([]int, numSWA),
		}
		for i := range numSWA {
			st.swaLastSeq[i] = -1
			if entry.StoredGroups&swaMasks[i] != 0 {
				st.swaCount[i] = 1
				if st.swaCount[i] >= info.SWAWindowBlocks[i] {
					st.swaLastSeq[i] = 0
				}
			}
		}
		states[entry.PodIdentifier] = st
	}

	for b := 1; b < len(keys); b++ {
		if len(states) == 0 {
			break
		}

		podGroups := make(map[string]uint32)
		for _, entry := range keyToPods[keys[b]] {
			podGroups[entry.PodIdentifier] |= entry.StoredGroups
		}

		for podID, st := range states {
			groups, present := podGroups[podID]

			if st.fullActive {
				if present && groups&fullMask != 0 {
					st.fullLastSeq = b
				} else {
					st.fullActive = false
				}
			}

			for i := range numSWA {
				if present && groups&swaMasks[i] != 0 {
					st.swaCount[i]++
					if st.swaCount[i] >= info.SWAWindowBlocks[i] {
						st.swaLastSeq[i] = b
					}
				} else {
					st.swaCount[i] = 0
				}
			}
		}
	}

	scores := make(map[string]float64)
	for podID, st := range states {
		checkpoint := st.fullLastSeq
		allMet := true
		for i := range numSWA {
			if st.swaLastSeq[i] < 0 {
				allMet = false
				break
			}
			if st.swaLastSeq[i] < checkpoint {
				checkpoint = st.swaLastSeq[i]
			}
		}
		if !allMet {
			continue
		}
		scores[podID] = float64(checkpoint + 1)
	}

	return scores
}