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
	"fmt"
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
	// extraFeatures provides per-block multimodal data that taints the hash;
	// nil means text-only (no taint). When non-nil, its length must match the
	// number of token chunks.
	// It returns a slice of generated Keys.
	TokensToKVBlockKeys(
		parentKey BlockHash, tokens []uint32, modelName string,
		extraFeatures []*BlockExtraFeatures,
	) ([]BlockHash, error)

	// BlockSize returns the number of tokens per block used by this processor.
	BlockSize() int
}

// chunkedTokenDatabase is a concrete implementation of TokenDatabase.
// It mimics the chunkedTokenDatabase in the Python code.
type chunkedTokenDatabase struct {
	TokenProcessorConfig
	encoder cbor.EncMode // cached CBOR encoder for interoperable encoding
}

var _ TokenProcessor = &chunkedTokenDatabase{}

// NewChunkedTokenDatabase creates a new instance with the given config and metadata.
func NewChunkedTokenDatabase(config *TokenProcessorConfig) (TokenProcessor, error) {
	if config == nil {
		config = DefaultTokenProcessorConfig()
	}

	if config.BlockSize <= 0 {
		return nil, fmt.Errorf("blockSize must be greater than 0, got %d", config.BlockSize)
	}

	if config.initHash == 0 {
		// Create initial hash
		h := fnv.New64a()
		_, _ = h.Write([]byte(config.HashSeed))
		config.initHash = h.Sum64()
	}

	encoder, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		return nil, fmt.Errorf("failed to create CBOR encoder: %w", err)
	}

	return &chunkedTokenDatabase{
		TokenProcessorConfig: *config,
		encoder:              encoder,
	}, nil
}

// getInitHash returns the initial hash for the given model name.
func (db *chunkedTokenDatabase) getInitHash(modelName string) uint64 {
	return db.hash(db.initHash, nil, modelName)
}

// hash computes the uint64 FNV-64a hash of the given parent, tokens,
// and extra keys.
//
// The hash is computed using FNV-64a over the CBOR canonical encoding of
// [parent, tokens, extra], ensuring deterministic results across runs and
// compatibility with vLLM's prefix caching algorithm.
//
// The extra parameter enables cache differentiation for LoRA adapters and
// multi-modal content. Supported types: nil, int, string, map[string]interface{}.
// Must be CBOR-serializable.
func (db *chunkedTokenDatabase) hash(parent uint64, tokens []uint32, extra interface{}) uint64 {
	payload := []interface{}{parent, tokens, extra}

	b, err := db.encoder.Marshal(payload)
	if err != nil {
		log.FromContext(context.Background()).Error(err, "failed to marshal payload to CBOR")
		return 0
	}

	h := fnv.New64a()
	_, _ = h.Write(b)
	return h.Sum64()
}

// prefixHashes returns a slice of uint64 hashes.
// extraFeatures must be the same length as tokenChunks (callers guarantee this).
func (db *chunkedTokenDatabase) prefixHashes(
	parentHash uint64, tokenChunks [][]uint32, extraFeatures []*BlockExtraFeatures,
) []uint64 {
	prefix := parentHash
	hashes := make([]uint64, len(tokenChunks))
	for i, chunk := range tokenChunks {
		var extra interface{}
		if extraFeatures[i] != nil {
			extra = extraFeatures[i].MMHashes
		}
		prefix = db.hash(prefix, chunk, extra)
		hashes[i] = prefix
	}
	return hashes
}

// BlockSize returns the number of tokens per block.
func (db *chunkedTokenDatabase) BlockSize() int {
	return db.TokenProcessorConfig.BlockSize
}

// chunkTokens splits the input slice of tokens into chunks of size blockSize.
func (db *chunkedTokenDatabase) chunkTokens(tokens []uint32) [][]uint32 {
	bs := db.TokenProcessorConfig.BlockSize
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += bs {
		end := i + bs
		if end > len(tokens) {
			break // no partial blocks
		}

		chunks = append(chunks, tokens[i:end])
	}

	return chunks
}

// TokensToKVBlockKeys converts tokens into kv_block.Keys.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(
	parentKey BlockHash, tokens []uint32, modelName string,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHash, error) {
	var currentParentHash uint64
	if parentKey != EmptyBlockHash {
		currentParentHash = uint64(parentKey)
	} else {
		currentParentHash = db.getInitHash(modelName)
	}

	chunks := db.chunkTokens(tokens)
	if len(chunks) == 0 {
		return nil, nil
	}

	if extraFeatures == nil {
		extraFeatures = make([]*BlockExtraFeatures, len(chunks))
	} else if len(extraFeatures) != len(chunks) {
		return nil, fmt.Errorf("extraFeatures length %d does not match token chunk count %d (blockSize=%d, tokens=%d)",
			len(extraFeatures), len(chunks), db.TokenProcessorConfig.BlockSize, len(tokens))
	}

	ph := db.prefixHashes(currentParentHash, chunks, extraFeatures)

	return utils.SliceMap(ph, func(hashVal uint64) BlockHash {
		return BlockHash(hashVal)
	}), nil
}
