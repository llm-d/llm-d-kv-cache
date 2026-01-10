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

package kvblock

import (
	"context"
	"hash/fnv"

	"github.com/fxamacker/cbor/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
)

// defaultBlockSize is the default number of tokens per block.
// 16 is the default value used by vLLM.
const defaultBlockSize = 16

// TokenProcessorConfig holds the configuration for the token processor.
type TokenProcessorConfig struct {
	BlockSize int `json:"blockSize"`
	// HashSeed is used to prefix initial hash chunks, similarly to vLLM's NONE_HASH.
	// This should be aligned with vLLM's `PYTHONHASHSEED` environment variable.
	// The system's deployer is responsible for aligning the vLLM deployments
	// with the same seed value.
	HashSeed string `json:"hashSeed"`
	initHash uint64 // cache once
}

// DefaultTokenProcessorConfig returns the default configuration for the token processor.
func DefaultTokenProcessorConfig() *TokenProcessorConfig {
	return &TokenProcessorConfig{
		BlockSize: defaultBlockSize,
		HashSeed:  "",
	}
}

// TokenProcessor defines the interface for converting tokens to
// KVBlockKeys.
type TokenProcessor interface {
	// TokensToKVBlockKeys converts tokens into kv_block.Keys.
	// It accepts an optional parentKey to continue a hash chain.
	// It returns a slice of generated Keys.
	TokensToKVBlockKeys(parentKey BlockHash, tokens []uint32, modelName string) []BlockHash
}

// chunkedTokenDatabase is a concrete implementation of TokenDatabase.
// It mimics the chunkedTokenDatabase in the Python code.
type chunkedTokenDatabase struct {
	TokenProcessorConfig
}

var _ TokenProcessor = &chunkedTokenDatabase{}

// NewChunkedTokenDatabase creates a new instance with the given config and metadata.
func NewChunkedTokenDatabase(config *TokenProcessorConfig) TokenProcessor {
	if config == nil {
		config = DefaultTokenProcessorConfig()
	} // TODO: validate?

	if config.initHash == 0 {
		// Create initial hash
		h := fnv.New64a()
		_, _ = h.Write([]byte(config.HashSeed))
		config.initHash = h.Sum64()
	}

	return &chunkedTokenDatabase{
		TokenProcessorConfig: *config,
	}
}

// getInitHash returns the initial hash for the given model name.
func (db *chunkedTokenDatabase) getInitHash(modelName string) uint64 {
	return db.hash(db.initHash, nil, modelName)
}

// hash computes the uint64 FNV-64a hash of the given parent, tokens,
// and extra keys.
func (db *chunkedTokenDatabase) hash(parent uint64, tokens []uint32, extra interface{}) uint64 {
	payload := []interface{}{parent, tokens, extra}

	encMode, err := cbor.CanonicalEncOptions().EncMode() // deterministic
	if err != nil {
		log.FromContext(context.Background()).Error(err, "failed to create CBOR encoder")
		return 0
	}

	b, err := encMode.Marshal(payload)
	if err != nil {
		log.FromContext(context.Background()).Error(err, "failed to marshal payload to CBOR")
		return 0
	}

	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

// prefixHashes returns a slice of uint64 hashes.
func (db *chunkedTokenDatabase) prefixHashes(parentHash uint64, tokenChunks [][]uint32) []uint64 {
	prefix := parentHash
	hashes := make([]uint64, len(tokenChunks))
	for i, chunk := range tokenChunks {
		prefix = db.hash(prefix, chunk, nil)
		hashes[i] = prefix
	}
	return hashes
}

// chunkTokens splits the input slice of tokens into chunks of size chunkSize.
func (db *chunkedTokenDatabase) chunkTokens(tokens []uint32) [][]uint32 {
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += db.BlockSize {
		end := i + db.BlockSize
		if end > len(tokens) {
			break // no partial blocks
		}

		chunks = append(chunks, tokens[i:end])
	}

	return chunks
}

// TokensToKVBlockKeys converts tokens into kv_block.Keys.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(parentKey BlockHash, tokens []uint32, modelName string) []BlockHash {
	var currentParentHash uint64
	if parentKey != EmptyBlockHash {
		currentParentHash = uint64(parentKey)
	} else {
		currentParentHash = db.getInitHash(modelName)
	}

	chunks := db.chunkTokens(tokens)
	if len(chunks) == 0 {
		return nil
	}

	ph := db.prefixHashes(currentParentHash, chunks)

	return utils.SliceMap(ph, func(hashVal uint64) BlockHash {
		return BlockHash(hashVal)
	})
}
