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
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/metrics"
	"k8s.io/apimachinery/pkg/util/sets"
)

// IndexConfig holds the configuration for the KV-block index.
// It may configure several backends such as listed within the struct.
// If multiple backends are configured, only the first one will be used.
type IndexConfig struct {
	// InMemoryConfig holds the configuration for the in-memory index.
	InMemoryConfig *InMemoryIndexConfig `json:"inMemoryConfig"`
	// RedisConfig holds the configuration for the Redis index.
	RedisConfig *RedisIndexConfig `json:"redisConfig"`
	// ValkeyConfig holds the configuration for the Valkey index.
	ValkeyConfig *RedisIndexConfig `json:"valkeyConfig"`
	// CostAwareMemoryConfig holds the configuration for the cost-aware memory index.
	CostAwareMemoryConfig *CostAwareMemoryIndexConfig `json:"costAwareMemoryConfig"`

	// EnableMetrics toggles whether admissions/evictions/hits/misses are
	// recorded.
	EnableMetrics bool `json:"enableMetrics"`
	// MetricsLoggingInterval defines the interval at which metrics are logged.
	// If zero, metrics logging is disabled.
	// Requires `EnableMetrics` to be true.
	MetricsLoggingInterval time.Duration `json:"metricsLoggingInterval"`
}

// DefaultIndexConfig returns a default configuration for the KV-block index.
func DefaultIndexConfig() *IndexConfig {
	return &IndexConfig{
		InMemoryConfig: DefaultInMemoryIndexConfig(),
		EnableMetrics:  false,
	}
}

// NewIndex creates a new Index instance.
func NewIndex(ctx context.Context, cfg *IndexConfig) (Index, error) {
	if cfg == nil {
		cfg = DefaultIndexConfig()
	}

	var idx Index
	var err error

	switch {
	case cfg.CostAwareMemoryConfig != nil:
		idx, err = NewCostAwareMemoryIndex(cfg.CostAwareMemoryConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create cost-aware memory index: %w", err)
		}
	case cfg.ValkeyConfig != nil:
		//nolint:contextcheck // NewValkeyIndex does not accept context parameter
		idx, err = NewValkeyIndex(cfg.ValkeyConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create Valkey index: %w", err)
		}
	case cfg.RedisConfig != nil:
		//nolint:contextcheck // NewRedisIndex does not accept context parameter
		idx, err = NewRedisIndex(cfg.RedisConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create Redis index: %w", err)
		}
	case cfg.InMemoryConfig != nil:
		idx, err = NewInMemoryIndex(cfg.InMemoryConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create in-memory index: %w", err)
		}
	default:
		return nil, fmt.Errorf("no valid index configuration provided")
	}

	// wrap in metrics only if enabled
	if cfg.EnableMetrics {
		idx = NewInstrumentedIndex(idx)
		metrics.Register()
		if cfg.MetricsLoggingInterval > 0 {
			// this is non-blocking
			metrics.StartMetricsLogging(ctx, cfg.MetricsLoggingInterval)
		}
	}

	return idx, nil
}

// Index defines the interface for a backend that manages KV-block
// indexing.
//
// An index backend is a data store that will aggregate possibly the entire
// global KV cache block index, and will be used to retrieve pod-localities
// for a given set of consecutive keys that constitute a prefix-cache hit.
// The hit may not necessarily be on all keys, but of the longest prefix match.
//
// The index backend allows efficient tracking of which vLLM engines hold which
// KV-blocks, on what device tier, and when they were last updated.
//
// Index operations are thread-safe and can be performed concurrently.
type Index interface {
	// Lookup receives a list of keys and a set of pod identifiers,
	// and retrieves the filtered pods associated with those keys.
	// The filtering is done based on the pod identifiers provided.
	// If the podIdentifierSet is empty, all pods are returned.
	//
	// It returns:
	// 1. A map where the keys are those in requestKeys and the values are pod-identifiers.
	// 2. An error if any occurred during the operation.
	Lookup(ctx context.Context, requestKeys []BlockHash, podIdentifierSet sets.Set[string]) (map[BlockHash][]PodEntry, error)
	// Add adds a set of engineKeys/requestKeys and their associated pod entries to the index backend.
	Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error
	// Evict removes an engineKey and its associated pod entries from the index backend.
	Evict(ctx context.Context, engineKey BlockHash, entries []PodEntry) error
	// GetRequestKey returns the requestKey associated with the given engineKey.
	GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error)
}

// BlockHash struct represents a unique identifier for a KV-cache block.
type BlockHash uint64

// EmptyBlockHash represents an invalid or uninitialized block hash.
// This serves as the "error value".
const EmptyBlockHash BlockHash = 0

// String returns a string representation of the Key.
func (c BlockHash) String() string {
	return fmt.Sprintf("%d", uint64(c))
}

// PodEntry struct represents a pod entry in the KV-block index.
type PodEntry struct {
	// PodIdentifier is the unique identifier for the pod.
	PodIdentifier string
	// DeviceTier is the tier of the device where the KV-block is stored.
	DeviceTier string
}

// String returns a string representation of the PodEntry.
func (e *PodEntry) String() string {
	return fmt.Sprintf("%s@%s", e.PodIdentifier, e.DeviceTier)
}
