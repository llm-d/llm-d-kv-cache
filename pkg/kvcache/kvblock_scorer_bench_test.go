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

package kvcache_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// BenchmarkLongestPrefixScorer tracks the scoring hot path: every routed
// request pays one Score call. Scenarios:
//
//   - legacy: ungrouped entries, nil catalog — the non-HMA deployment; guards
//     the phase-1 cost (prefix walk + prefix-sum bookkeeping).
//   - hma_full_hit: hybrid entries, everything cached — phase 2 terminates
//     after one trailing window; the common warm-cache HMA case.
//   - hma_swa_absent: hybrid entries with the SWA group evicted — phase 2
//     scans the entire prefix and collapses the hit; the worst-case scan.
func BenchmarkLongestPrefixScorer(b *testing.B) {
	const (
		numPods   = 32
		numBlocks = 512
		window    = 1024 // tokens; need = cdiv(1023, 16) = 64 trailing blocks
	)

	scenarios := []struct {
		name string
		hma  bool // grouped entries + catalog
		swa  bool // include sliding-window group entries
	}{
		{name: "legacy", hma: false},
		{name: "hma_full_hit", hma: true, swa: true},
		{name: "hma_swa_absent", hma: true, swa: false},
	}

	for _, sc := range scenarios {
		keys := make([]kvblock.BlockHash, numBlocks)
		keyToPods := make(map[kvblock.BlockHash][]kvblock.PodEntry, numBlocks)
		for i := range keys {
			keys[i] = kvblock.BlockHash(i + 1) // #nosec G115 -- test data, i is small
			entries := make([]kvblock.PodEntry, 0, numPods*2)
			for p := 0; p < numPods; p++ {
				pod := fmt.Sprintf("pod-%d", p)
				if !sc.hma {
					entries = append(entries, kvblock.PodEntry{PodIdentifier: pod, DeviceTier: "gpu"})
					continue
				}
				entries = append(entries, hmaEntry(pod, 0))
				if sc.swa {
					entries = append(entries, hmaEntry(pod, 1))
				}
			}
			keyToPods[keys[i]] = entries
		}

		var catalog *kvblock.GroupCatalog
		if sc.hma {
			catalog = kvblock.NewGroupCatalog()
			w := window
			for p := 0; p < numPods; p++ {
				pod := fmt.Sprintf("pod-%d", p)
				catalog.Learn(pod, 0, kvblock.GroupMetadata{IsMainAttention: true, BlockSize: hmaTestBlockSize})
				catalog.Learn(pod, 1, kvblock.GroupMetadata{BlockSize: hmaTestBlockSize, SlidingWindowSize: &w})
			}
		}
		scorer := &kvcache.LongestPrefixScorer{
			MediumWeights:      map[string]float64{"gpu": 1.0, "cpu": 0.5},
			Catalog:            catalog,
			CanonicalBlockSize: hmaTestBlockSize,
		}

		b.Run(sc.name, func(b *testing.B) {
			ctx := context.Background()
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := scorer.Score(ctx, keys, keyToPods); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
