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

package kvblock_test

import (
	"fmt"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// --- ParseRawExtraKeys benchmarks ---

func BenchmarkParseRawExtraKeys_TextOnly(b *testing.B) {
	// 128 blocks, all nil (text-only prompt)
	raw := make([][]any, 128)
	b.ResetTimer()
	for range b.N {
		_, _ = kvblock.ParseRawExtraKeys(raw)
	}
}

func BenchmarkParseRawExtraKeys_AllMM(b *testing.B) {
	// 128 blocks, each with one MM entry (large image prompt)
	raw := make([][]any, 128)
	for i := range raw {
		raw[i] = []any{[]any{"hash_abc123", int64(15 - i*16)}}
	}
	b.ResetTimer()
	for range b.N {
		_, _ = kvblock.ParseRawExtraKeys(raw)
	}
}

func BenchmarkParseRawExtraKeys_Mixed(b *testing.B) {
	// 128 blocks: 64 nil + 64 MM (typical two-image prompt)
	raw := make([][]any, 128)
	for i := 64; i < 128; i++ {
		raw[i] = []any{[]any{"hash_img2", int64(5 - (i-64)*16)}}
	}
	b.ResetTimer()
	for range b.N {
		_, _ = kvblock.ParseRawExtraKeys(raw)
	}
}

// --- ComputeBlockExtraFeatures benchmarks ---

func BenchmarkComputeBlockExtraFeatures_SingleImage(b *testing.B) {
	mmHashes := map[string][]string{"image": {"hash_A"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 15, Length: 976}},
	}
	b.ResetTimer()
	for range b.N {
		_ = kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 1024)
	}
}

func BenchmarkComputeBlockExtraFeatures_TwoImages(b *testing.B) {
	mmHashes := map[string][]string{"image": {"hash_A", "hash_B"}}
	mmPlaceholders := map[string][]kvblock.PlaceholderRange{
		"image": {{Offset: 15, Length: 976}, {Offset: 993, Length: 864}},
	}
	b.ResetTimer()
	for range b.N {
		_ = kvblock.ComputeBlockExtraFeatures(mmHashes, mmPlaceholders, 16, 1872)
	}
}

func BenchmarkComputeBlockExtraFeatures_NoMM(b *testing.B) {
	b.ResetTimer()
	for range b.N {
		_ = kvblock.ComputeBlockExtraFeatures(nil, nil, 16, 2048)
	}
}

// --- TokensToKVBlockKeys benchmarks ---

func makeTokens(n int) []uint32 {
	t := make([]uint32, n)
	for i := range t {
		t[i] = uint32(i + 1)
	}
	return t
}

func BenchmarkTokensToKVBlockKeys_TextOnly(b *testing.B) {
	for _, numTokens := range []int{256, 1024, 4096} {
		b.Run(fmt.Sprintf("tokens=%d", numTokens), func(b *testing.B) {
			proc, _ := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
				BlockSize: 16, HashSeed: "bench",
			})
			tokens := makeTokens(numTokens)
			b.ResetTimer()
			for range b.N {
				_, _ = proc.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", nil)
			}
		})
	}
}

func BenchmarkTokensToKVBlockKeys_WithMM(b *testing.B) {
	for _, numTokens := range []int{256, 1024, 4096} {
		b.Run(fmt.Sprintf("tokens=%d", numTokens), func(b *testing.B) {
			proc, _ := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{
				BlockSize: 16, HashSeed: "bench",
			})
			tokens := makeTokens(numTokens)
			numBlocks := numTokens / 16
			features := make([]*kvblock.BlockExtraFeatures, numBlocks)
			// Image covers first half of blocks
			for i := range numBlocks / 2 {
				features[i] = &kvblock.BlockExtraFeatures{
					MMHashes: []kvblock.MMHash{{Hash: "img_hash", Offset: int64(-i * 16)}},
				}
			}
			b.ResetTimer()
			for range b.N {
				_, _ = proc.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens, "model", features)
			}
		})
	}
}
