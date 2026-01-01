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

package prefixstore

import (
	"encoding/binary"
	"fmt"

	"github.com/cespare/xxhash/v2"
)

// BlockHasher handles the computation of block hashes for chunked text data.
// It maintains state for sequential hash computation where each block's hash
// depends on the previous block's hash.
type BlockHasher struct {
	digest       *xxhash.Digest
	previousHash uint64
}

func NewBlockHasher() *BlockHasher {
	return &BlockHasher{
		digest:       xxhash.New(),
		previousHash: 0,
	}
}

// Reset resets the hasher state for a new sequence of blocks.
func (h *BlockHasher) Reset() {
	h.digest.Reset()
}

// ComputeBlockHash computes the hash for a block of text data.
// The hash depends on both the current block content and the previous block's hash.
func (h *BlockHasher) ComputeBlockHash(data []byte) (uint64, error) {
	h.digest.Reset()

	// Include previous hash to create a chain
	if err := binary.Write(h.digest, binary.LittleEndian, h.previousHash); err != nil {
		return 0, fmt.Errorf("failed to write previous hash: %w", err)
	}

	// Include current block data
	if _, err := h.digest.Write(data); err != nil {
		return 0, fmt.Errorf("failed to write block data: %w", err)
	}

	blockHash := h.digest.Sum64()
	h.previousHash = blockHash

	return blockHash, nil
}

// GetPreviousHash returns the current previous hash value.
func (h *BlockHasher) GetPreviousHash() uint64 {
	return h.previousHash
}
