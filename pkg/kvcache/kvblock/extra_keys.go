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

package kvblock

import (
	"fmt"
	"math"
	"sort"
)

// MMHash represents a single multimodal content hash entry with its block-relative offset.
// This matches vLLM's per-block extra_keys format: (content_hash, offset).
type MMHash struct {
	Hash   string
	Offset int64
}

// BlockExtraFeatures holds per-block extra data that taints the block hash.
// A nil *BlockExtraFeatures means pure text (no taint).
type BlockExtraFeatures struct {
	MMHashes []MMHash
}

// PlaceholderRange describes a contiguous range of placeholder tokens for one
// multimodal item within the full token sequence.
type PlaceholderRange struct {
	Offset int // absolute start token index
	Length int // number of placeholder tokens
}

// ParseRawExtraKeys converts the raw [][]any from BlockStoredEvent.ExtraKeys
// into typed []*BlockExtraFeatures. Each inner []any element is expected to be
// a 2-element [string, int] tuple representing (mm_hash, offset).
// nil inner slices produce nil entries. Returns nil if raw is nil.
func ParseRawExtraKeys(raw [][]any) ([]*BlockExtraFeatures, error) {
	if raw == nil {
		return nil, nil
	}

	result := make([]*BlockExtraFeatures, len(raw))
	for blockIdx, blockKeys := range raw {
		if blockKeys == nil {
			continue
		}

		features := &BlockExtraFeatures{}
		for entryIdx, entry := range blockKeys {
			tuple, ok := entry.([]any)
			if !ok {
				// Skip non-tuple entries (e.g. LoRA string pairs).
				continue
			}
			if len(tuple) != 2 {
				continue
			}

			hash, ok := tuple[0].(string)
			if !ok {
				continue
			}

			offset, err := asInt64(tuple[1])
			if err != nil {
				return nil, fmt.Errorf("extra_keys[%d][%d] offset: %w", blockIdx, entryIdx, err)
			}

			features.MMHashes = append(features.MMHashes, MMHash{Hash: hash, Offset: offset})
		}

		if len(features.MMHashes) > 0 {
			result[blockIdx] = features
		}
	}

	return result, nil
}

// mmItem flattens one multimodal placeholder with its content hash for sorting.
type mmItem struct {
	hash  string
	start int
	end   int
}

// ComputeBlockExtraFeatures converts tokenizer-provided multimodal metadata into
// per-block extra features, matching vLLM's generate_block_hash_extra_keys() algorithm.
//
// For each block, it finds overlapping placeholder ranges and computes
// (hash, block_relative_offset) entries. The block_relative_offset can be negative
// for continuation blocks where the placeholder started in a previous block.
func ComputeBlockExtraFeatures(
	mmHashes map[string][]string,
	mmPlaceholders map[string][]PlaceholderRange,
	blockSize, numTokens int,
) []*BlockExtraFeatures {
	if len(mmHashes) == 0 || blockSize <= 0 || numTokens <= 0 {
		return nil
	}

	// Flatten all placeholder ranges with their hashes, sorted by start position.
	var items []mmItem
	for modality, hashes := range mmHashes {
		ranges, ok := mmPlaceholders[modality]
		if !ok {
			continue
		}
		n := len(hashes)
		if len(ranges) < n {
			n = len(ranges)
		}
		for i := 0; i < n; i++ {
			items = append(items, mmItem{
				hash:  hashes[i],
				start: ranges[i].Offset,
				end:   ranges[i].Offset + ranges[i].Length,
			})
		}
	}

	if len(items) == 0 {
		return nil
	}

	sort.Slice(items, func(i, j int) bool { return items[i].start < items[j].start })

	numBlocks := numTokens / blockSize
	result := make([]*BlockExtraFeatures, numBlocks)

	for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
		blockStart := blockIdx * blockSize
		blockEnd := blockStart + blockSize

		var hashes []MMHash
		for _, item := range items {
			// Placeholder ends before this block — skip.
			if item.end <= blockStart {
				continue
			}
			// Placeholder starts at or after block end — no more overlaps for this block
			// (items are sorted).
			if item.start >= blockEnd {
				break
			}
			// Overlap: compute block-relative offset.
			hashes = append(hashes, MMHash{
				Hash:   item.hash,
				Offset: int64(item.start - blockStart),
			})
		}

		if len(hashes) > 0 {
			result[blockIdx] = &BlockExtraFeatures{MMHashes: hashes}
		}
	}

	return result
}

// asInt64 converts msgpack numeric types to int64.
func asInt64(raw any) (int64, error) {
	switch val := raw.(type) {
	case int8:
		return int64(val), nil
	case int16:
		return int64(val), nil
	case int32:
		return int64(val), nil
	case int64:
		return val, nil
	case uint8:
		return int64(val), nil
	case uint16:
		return int64(val), nil
	case uint32:
		return int64(val), nil
	case uint64:
		if val > math.MaxInt64 {
			return 0, fmt.Errorf("uint64 value %d overflows int64", val)
		}
		return int64(val), nil
	case int:
		return int64(val), nil
	default:
		return 0, fmt.Errorf("unsupported numeric type: %T", val)
	}
}
