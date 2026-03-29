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
	"strings"
	"sync"

	lru "github.com/hashicorp/golang-lru/v2"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	defaultInMemoryIndexSize = 1e8 // TODO: change to memory-size based configuration
	defaultPodsPerKey        = 10  // number of pods per key
)

// InMemoryIndexConfig holds the configuration for the InMemoryIndex.
type InMemoryIndexConfig struct {
	// Size is the maximum number of keys that can be stored in the index.
	Size int `json:"size"`
	// PodCacheSize is the maximum number of pod entries per key.
	PodCacheSize int `json:"podCacheSize"`
}

// DefaultInMemoryIndexConfig returns a default configuration for the InMemoryIndex.
func DefaultInMemoryIndexConfig() *InMemoryIndexConfig {
	return &InMemoryIndexConfig{
		Size:         defaultInMemoryIndexSize,
		PodCacheSize: defaultPodsPerKey,
	}
}

// NewInMemoryIndex creates a new InMemoryIndex instance.
func NewInMemoryIndex(cfg *InMemoryIndexConfig) (*InMemoryIndex, error) {
	if cfg == nil {
		cfg = DefaultInMemoryIndexConfig()
	}

	cache, err := lru.New[BlockHash, *PodCache](cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory index: %w", err)
	}

	engineToRequestKeys, err := lru.New[BlockHash, BlockHash](cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory engine key map: %w", err)
	}

	podToRequestKeys, err := lru.New[string, []engineKeyToRequestKeyMapping](cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize pod-to-request-key map: %w", err)
	}

	return &InMemoryIndex{
		data:                cache,
		engineToRequestKeys: engineToRequestKeys,
		podToRequestKeys:    podToRequestKeys,
		podCacheSize:        cfg.PodCacheSize,
	}, nil
}

// InMemoryIndex is an in-memory implementation of the Index interface.
type InMemoryIndex struct {
	// data holds the mapping of requestKeys to sets of pod identifiers.
	data *lru.Cache[BlockHash, *PodCache]
	// engineToRequestKeys holds the mapping of engineKeys to requestKeys.
	engineToRequestKeys *lru.Cache[BlockHash, BlockHash]
	// podToRequestKeys is a reverse index: podIdentifier -> []engineKeyToRequestKeyMapping.
	podToRequestKeys *lru.Cache[string, []engineKeyToRequestKeyMapping]
	// podCacheSize is the maximum number of pod entries per key.
	podCacheSize int
}

var _ Index = &InMemoryIndex{}

// PodCache represents a cache for pod entries.
type PodCache struct {
	// cache is an LRU cache that maps PodEntry to their last access time.
	// thread-safe.
	cache *lru.Cache[PodEntry, struct{}]
	// mu protects the cache from concurrent access during check-and-set operations.
	mu sync.Mutex
}

// Lookup receives a list of requestKeys and a set of pod identifiers,
// and retrieves the filtered pods associated with those keys.
// The filtering is done based on the pod identifiers provided.
// If the podIdentifierSet is empty, all pods are returned.
//
// It returns:
// 1. A map where the keys are those in (1) and the values are pod-identifiers.
// 2. An error if any occurred during the operation.
func (m *InMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no requestKeys provided for lookup")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	highestHitIdx := 0

	for idx, requestKey := range requestKeys {
		if pods, found := m.data.Get(requestKey); found { //nolint:nestif // TODO: can this be optimized?
			if pods == nil || pods.cache.Len() == 0 {
				traceLogger.Info("no pods found for key, cutting search", "key", requestKey)
				return podsPerKey, nil // early stop since prefix-chain breaks here
			}

			highestHitIdx = idx

			if podIdentifierSet.Len() == 0 {
				// If no pod identifiers are provided, return all pods
				podsPerKey[requestKey] = pods.cache.Keys()
			} else {
				// Filter pods based on the provided pod identifiers
				for _, pod := range pods.cache.Keys() {
					if podIdentifierSet.Has(pod.PodIdentifier) {
						podsPerKey[requestKey] = append(podsPerKey[requestKey], pod)
					}
				}
			}
		} else {
			traceLogger.Info("key not found in index", "key", requestKey)
		}
	}

	traceLogger.Info("lookup completed", "highest-hit-index", highestHitIdx,
		"pods-per-key", podsPerKeyPrintHelper(podsPerKey))

	return podsPerKey, nil
}

// Add adds a set of engineKeys/requestKeys and their associated pod entries to the index backend.
// If engineKeys is nil, only requestKey -> PodEntry mappings are created (no engineKey -> requestKey mapping).
// This is used for speculative entries where engine keys are not yet known.
func (m *InMemoryIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}
	if engineKeys != nil && len(engineKeys) != len(requestKeys) {
		return fmt.Errorf("mismatch between engine keys and request keys length")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Add")

	for i, requestKey := range requestKeys {
		// 1. Store engineKey -> requestKey mapping (only if engineKeys provided)
		curEngineKey := EmptyBlockHash
		if engineKeys != nil {
			m.engineToRequestKeys.Add(engineKeys[i], requestKey)
		}

		// 2. Store requestKey -> PodCache mapping
		var podCache *PodCache
		var found bool

		// Try to get existing cache first
		podCache, found = m.data.Get(requestKey)
		//nolint:nestif // double-checked locking pattern
		if !found {
			// Create new cache
			cache, err := lru.New[PodEntry, struct{}](m.podCacheSize)
			if err != nil {
				return fmt.Errorf("failed to create pod cache for key %s: %w", requestKey.String(), err)
			}

			newPodCache := &PodCache{
				cache: cache,
			}

			// Try to add, but use existing if another thread added it first
			// This is a bounded retry (1) - not perfectly safe but for practical use-cases and scenarios
			// this should be sufficient
			contains, _ := m.data.ContainsOrAdd(requestKey, newPodCache)
			if contains {
				podCache, found = m.data.Get(requestKey)
				if !found { // Extremely irregular workload pattern - key evicted
					m.data.Add(requestKey, newPodCache)
					podCache = newPodCache
				}
			} else {
				// We successfully added our cache
				podCache = newPodCache
			}
		}

		podCache.mu.Lock()
		for _, entry := range entries {
			podCache.cache.Add(entry, struct{}{})

			// 3. Maintain reverse index: podIdentifier -> []engineKeyToRequestKeyMapping
			if existing, ok := m.podToRequestKeys.Peek(entry.PodIdentifier); ok {
				found := false
				for _, rk := range existing {
					if rk.requestKey == requestKey {
						found = true
						break
					}
				}
				if !found {
					m.podToRequestKeys.Add(entry.PodIdentifier, append(existing, engineKeyToRequestKeyMapping{requestKey: requestKey, engineKey: curEngineKey}))
				}
			} else {
				m.podToRequestKeys.Add(entry.PodIdentifier, []engineKeyToRequestKeyMapping{{requestKey: requestKey, engineKey: curEngineKey}})
			}
		}
		podCache.mu.Unlock()

		traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", entries)
	}

	return nil
}

// Evict removes a key and its associated pod entries from the index backend.
// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
// or a RequestKey (used directly for speculative entries without engineKey mapping).
func (m *InMemoryIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Evict")

	var requestKey BlockHash
	hasEngineKeyMapping := false

	switch keyType {
	case EngineKey:
		rk, found := m.engineToRequestKeys.Get(key)
		if !found {
			traceLogger.Info("engineKey not found in mapping, nothing to evict", "engineKey", key)
			return nil
		}
		requestKey = rk
		hasEngineKeyMapping = true
	case RequestKey:
		requestKey = key
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}

	podCache, found := m.data.Get(requestKey)
	if !found || podCache == nil {
		if hasEngineKeyMapping {
			traceLogger.Info("requestKey not found in index, cleaning up engineKey", "requestKey", requestKey, "engineKey", key)
			m.engineToRequestKeys.Remove(key)
		} else {
			traceLogger.Info("key not found in index, nothing to evict", "key", key)
		}
		return nil
	}

	podCache.mu.Lock()
	for _, entry := range entries {
		podCache.cache.Remove(entry)

		if rks, ok := m.podToRequestKeys.Peek(entry.PodIdentifier); ok {
			updated := rks[:0]
			for _, rk := range rks {
				if rk.requestKey != requestKey {
					updated = append(updated, rk)
				}
			}
			if len(updated) == 0 {
				m.podToRequestKeys.Remove(entry.PodIdentifier)
			} else {
				m.podToRequestKeys.Add(entry.PodIdentifier, updated)
			}
		}
	}

	isEmpty := podCache.cache.Len() == 0
	podCache.mu.Unlock()

	traceLogger.Info("evicted pods from key", "requestKey", requestKey, "key", key, "keyType", keyType, "pods", entries)

	// Remove key from main cache if empty.
	// Re-fetch and hold the lock through removal to prevent racing with Add.
	if !isEmpty {
		return nil
	}

	currentCache, stillExists := m.data.Get(requestKey)
	if !stillExists || currentCache == nil {
		return nil
	}

	currentCache.mu.Lock()
	if currentCache.cache.Len() == 0 {
		m.data.Remove(requestKey)
		if hasEngineKeyMapping {
			m.engineToRequestKeys.Remove(key)
		}
		traceLogger.Info("removed requestKey from index as no pods remain", "requestKey", requestKey, "key", key)
	}
	currentCache.mu.Unlock()

	return nil
}

// GetRequestKey returns the requestKey associated with the given engineKey.
// Returns an error if the engineKey mapping is missing (e.g., already evicted).
func (m *InMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	requestKey, found := m.engineToRequestKeys.Get(engineKey)
	if !found {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return requestKey, nil
}

// podsPerKeyPrintHelper formats a map of keys to pod names for printing.
func podsPerKeyPrintHelper(ks map[BlockHash][]PodEntry) string {
	var b strings.Builder
	for k, v := range ks {
		fmt.Fprintf(&b, "%s: %v\n", k.String(), utils.SliceMap(v, func(pod PodEntry) string {
			return pod.String()
		}))
	}
	return b.String()
}

// Clear removes all entries for the given podEntry from the index.
func (m *InMemoryIndex) Clear(ctx context.Context, podEntry PodEntry) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.InMemoryIndex.Clear")
	// Remove all entries for the given pod identifier
	entries, found := m.podToRequestKeys.Get(podEntry.PodIdentifier)
	if !found {
		traceLogger.Info("pod not found in reverse index, nothing to clear", "podEntry", podEntry)
		return nil
	}

	for _, entry := range entries {
		cache, exists := m.data.Peek(entry.requestKey)
		if !exists || cache == nil {
			continue
		}

		cache.mu.Lock()
		for _, entry := range cache.cache.Keys() {
			if entry.PodIdentifier != podEntry.PodIdentifier {
				continue
			}
			if podEntry.DeviceTier != "" && entry.DeviceTier != podEntry.DeviceTier {
				continue
			}
			cache.cache.Remove(entry)
		}
		isEmpty := cache.cache.Len() == 0
		cache.mu.Unlock()

		if isEmpty {
			m.data.Remove(entry.requestKey)
			if entry.engineKey != EmptyBlockHash {
				if _, exists := m.engineToRequestKeys.Peek(entry.engineKey); exists {
					m.engineToRequestKeys.Remove(entry.engineKey)
				}
			}
		}
	}

	if podEntry.DeviceTier == "" {
		m.podToRequestKeys.Remove(podEntry.PodIdentifier)
	} else {
		var remaining []engineKeyToRequestKeyMapping
		for _, entry := range entries {
			if exists := m.data.Contains(entry.requestKey); exists {
				remaining = append(remaining, entry)
			}
		}
		if len(remaining) == 0 {
			m.podToRequestKeys.Remove(podEntry.PodIdentifier)
		} else {
			m.podToRequestKeys.Add(podEntry.PodIdentifier, remaining)
		}
	}

	traceLogger.Info("Cleared pod entries from InMemoryIndex", "podEntry", podEntry)
	return nil
}
