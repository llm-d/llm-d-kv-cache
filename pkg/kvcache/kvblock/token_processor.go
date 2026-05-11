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
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash/fnv"

	"github.com/fxamacker/cbor/v2"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
)

// HashAlgorithm selects the hashing algorithm used by TokensToKVBlockKeys.
type HashAlgorithm string

const (
	// HashAlgorithmSHA256CBOR uses SHA256-CBOR hashing, matching vLLM's engine-key
	// computation.
	HashAlgorithmSHA256CBOR HashAlgorithm = "sha256_cbor"

	// HashAlgorithmFNV64a uses FNV-64a hashing over CBOR-encoded payloads.
	// This is the default algorithm.
	HashAlgorithmFNV64a HashAlgorithm = "fnv64a_cbor"
)

const (
	// defaultBlockSize is the default number of tokens per block (vLLM default).
	defaultBlockSize = 16
)

// TokenProcessorConfig holds the configuration for the token processor.
type TokenProcessorConfig struct {
	// BlockSize is deprecated. Use BlockSizeTokens instead.
	//
	// Deprecated: Use BlockSizeTokens instead.
	BlockSize int `json:"blockSize,omitempty"`

	// BlockSizeTokens is the number of tokens per block.
	// A value of zero is treated as "not set" and resolved to the default (16).
	// This applies to both FNV64a and SHA256-CBOR algorithms.
	BlockSizeTokens int `json:"blockSizeTokens"`

	// HashSeed is used to prefix initial hash chunks.
	// - For FNV64a: used directly as the initial hash seed
	// - For SHA256-CBOR: aligns with vLLM's PYTHONHASHSEED environment variable for NONE_HASH
	// The system's deployer is responsible for aligning the vLLM deployments
	// with the same seed value.
	HashSeed string `json:"hashSeed"`

	// HashAlgorithm selects the hashing algorithm.
	// Valid values: "fnv64a_cbor" (default), "sha256_cbor".
	HashAlgorithm HashAlgorithm `json:"hashAlgorithm,omitempty"`

	initHash uint64 // cache once for FNV
}

// DefaultTokenProcessorConfig returns the default configuration for the token processor.
func DefaultTokenProcessorConfig() *TokenProcessorConfig {
	return &TokenProcessorConfig{
		BlockSizeTokens: defaultBlockSize,
		HashSeed:        "",
		HashAlgorithm:   HashAlgorithmFNV64a,
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
	var cfg TokenProcessorConfig
	if config == nil {
		cfg = *DefaultTokenProcessorConfig()
	} else {
		cfg = *config // local copy — caller's struct is never mutated
	}

	// Default HashAlgorithm to FNV64a.
	if cfg.HashAlgorithm == "" {
		cfg.HashAlgorithm = HashAlgorithmFNV64a
	}

	// Apply defaults for omitted fields so partial configs (e.g. only hashSeed set) work correctly.
	if cfg.BlockSizeTokens == 0 && cfg.BlockSize == 0 {
		cfg.BlockSizeTokens = defaultBlockSize
	}

	// Handle backward compatibility: if only deprecated BlockSize is set, promote it.
	if cfg.BlockSizeTokens == 0 && cfg.BlockSize > 0 {
		cfg.BlockSizeTokens = cfg.BlockSize
	}

	if cfg.BlockSizeTokens <= 0 {
		// Report the actual invalid value the caller set, not the zero from the other field.
		invalidBlockSize := cfg.BlockSizeTokens
		if cfg.BlockSizeTokens == 0 && cfg.BlockSize != 0 {
			invalidBlockSize = cfg.BlockSize
		}
		return nil, fmt.Errorf("blockSizeTokens must be greater than 0, got %d", invalidBlockSize)
	}

	// Compute and cache FNV initHash only if using FNV algorithm.
	if cfg.HashAlgorithm == HashAlgorithmFNV64a && cfg.initHash == 0 {
		h := fnv.New64a()
		_, _ = h.Write([]byte(cfg.HashSeed))
		cfg.initHash = h.Sum64()
	}

	encoder, err := cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		return nil, fmt.Errorf("failed to create CBOR encoder: %w", err)
	}

	return &chunkedTokenDatabase{
		TokenProcessorConfig: cfg,
		encoder:              encoder,
	}, nil
}

// BlockSize returns the configured number of tokens per block.
func (db *chunkedTokenDatabase) BlockSize() int {
	return db.BlockSizeTokens
}

// TokensToKVBlockKeys converts tokens to block keys using the configured algorithm.
func (db *chunkedTokenDatabase) TokensToKVBlockKeys(
	parentKey BlockHash, tokens []uint32, modelName string,
	extraFeatures []*BlockExtraFeatures,
) ([]BlockHash, error) {
	switch db.HashAlgorithm {
	case HashAlgorithmFNV64a:
		return db.tokensToKVBlockKeysFNV(parentKey, tokens, modelName, extraFeatures)
	default: // HashAlgorithmSHA256CBOR and any unrecognised value
		return db.tokensToKVBlockKeysSHA256(parentKey, tokens, extraFeatures)
	}
}

// --- SHA256-CBOR path ---

// tokensToKVBlockKeysSHA256 implements the SHA256-CBOR hashing path, matching
// vLLM's engine-key computation with full support for extra features (LoRA,
// multimodal hashes, cache salt, prompt embeds). modelName is not included
// in the hash (vLLM sha256_cbor behaviour).
func (db *chunkedTokenDatabase) tokensToKVBlockKeysSHA256(
	parentKey BlockHash, tokens []uint32, extraFeatures []*BlockExtraFeatures,
) ([]BlockHash, error) {
	if parentKey != EmptyBlockHash {
		klog.FromContext(context.Background()).Error(
			fmt.Errorf("non-empty parentKey unsupported for sha256_cbor"),
			"TokensToKVBlockKeys: parentKey must be EmptyBlockHash for sha256_cbor",
			"parentKey", parentKey,
		)
		return nil, nil
	}

	chunks := chunkTokensSHA256(tokens, db.BlockSizeTokens)
	if len(chunks) == 0 {
		return nil, nil
	}

	// Validate or initialize extraFeatures to match chunk count
	if extraFeatures == nil {
		extraFeatures = make([]*BlockExtraFeatures, len(chunks))
	} else if len(extraFeatures) != len(chunks) {
		return nil, fmt.Errorf("extraFeatures length %d does not match token chunk count %d (blockSizeTokens=%d, tokens=%d)",
			len(extraFeatures), len(chunks), db.BlockSizeTokens, len(tokens))
	}

	initDigest := getInitHashSHA256(db.HashSeed)
	hashes := prefixHashesSHA256(initDigest, chunks, extraFeatures)

	return utils.SliceMap(hashes, func(h []byte) BlockHash {
		return BlockHash(blockHashToEngineKey(h))
	}), nil
}

// --- FNV-64a path ---

// tokensToKVBlockKeysFNV implements the original FNV-64a hashing path.
func (db *chunkedTokenDatabase) tokensToKVBlockKeysFNV(
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
		return nil, fmt.Errorf("extraFeatures length %d does not match token chunk count %d (blockSizeTokens=%d, tokens=%d)",
			len(extraFeatures), len(chunks), db.BlockSizeTokens, len(tokens))
	}

	ph := db.prefixHashes(currentParentHash, chunks, extraFeatures)

	return utils.SliceMap(ph, func(hashVal uint64) BlockHash {
		return BlockHash(hashVal)
	}), nil
}

// getInitHash returns the FNV-64a initial hash for the given model name.
func (db *chunkedTokenDatabase) getInitHash(modelName string) uint64 {
	return db.hash(db.initHash, nil, modelName)
}

// hash computes a FNV-64a hash over the CBOR encoding of [parent, tokens, extra].
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

// prefixHashes returns FNV-64a prefix hashes for the given chunks.
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

// chunkTokens splits tokens into chunks of db.BlockSizeTokens (FNV path).
func (db *chunkedTokenDatabase) chunkTokens(tokens []uint32) [][]uint32 {
	bs := db.BlockSizeTokens
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += bs {
		end := i + bs
		if end > len(tokens) {
			break
		}
		chunks = append(chunks, tokens[i:end])
	}
	return chunks
}

// --- SHA256-CBOR helpers (package-level, used by scorer and prefetch handler) ---

var cborEncMode cbor.EncMode // initialized once

func init() {
	var err error
	cborEncMode, err = cbor.CanonicalEncOptions().EncMode()
	if err != nil {
		panic(fmt.Sprintf("failed to create CBOR encoder: %v", err))
	}
}

// getInitHashSHA256 computes the 32-byte seed hash using SHA256-CBOR,
// matching vLLM's NONE_HASH initialization.
func getInitHashSHA256(seed string) []byte {
	b, err := cborEncMode.Marshal(seed)
	if err != nil {
		klog.FromContext(context.Background()).Error(err, "failed to marshal seed to CBOR")
		return nil
	}
	sum := sha256.Sum256(b)
	return sum[:]
}

// hashSHA256 hashes one block: SHA256(CBOR([parent, tokens, extra])),
// matching vLLM's prefix-cache hash chain. The extra parameter can be any
// type (nil, string, []any, etc.) and is serialized as-is by CBOR.
func hashSHA256(parent []byte, tokens []uint32, extra any) []byte {
	payload := []any{parent, tokens, extra}
	b, err := cborEncMode.Marshal(payload)
	if err != nil {
		klog.FromContext(context.Background()).Error(err, "failed to marshal payload to CBOR")
		return nil
	}
	sum := sha256.Sum256(b)
	return sum[:]
}

// prefixHashesSHA256 chains SHA256-CBOR hashes across token chunks,
// carrying the full 32-byte digest between blocks. Extras from BlockExtraFeatures
// are included in each block's hash via the CborExtras() method.
func prefixHashesSHA256(parentHash []byte, tokenChunks [][]uint32, extraFeatures []*BlockExtraFeatures) [][]byte {
	prefix := parentHash
	hashes := make([][]byte, len(tokenChunks))
	for i, chunk := range tokenChunks {
		var extra any
		if extraFeatures[i] != nil {
			extra = extraFeatures[i].CborExtras()
		}
		prefix = hashSHA256(prefix, chunk, extra)
		hashes[i] = prefix
	}
	return hashes
}

// chunkTokensSHA256 splits tokens into full chunks of exactly blockSize;
// trailing partial chunks are dropped (matches vLLM behaviour).
func chunkTokensSHA256(tokens []uint32, blockSize int) [][]uint32 {
	var chunks [][]uint32
	for i := 0; i < len(tokens); i += blockSize {
		end := i + blockSize
		if end > len(tokens) {
			break
		}
		chunks = append(chunks, tokens[i:end])
	}
	return chunks
}

// blockHashToEngineKey extracts the engine key from a SHA256 digest by taking
// the last 8 bytes (bytes 24–31), matching vLLM's truncation.
func blockHashToEngineKey(h []byte) uint64 {
	return binary.BigEndian.Uint64(h[24:])
}
