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
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/dgraph-io/ristretto/v2"
	"github.com/dustin/go-humanize"
	lru "github.com/hashicorp/golang-lru/v2"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	defaultNumCounters = 1e8 // 100M keys
	defaultBufferItems = 64  // default buffer size for ristretto
)

// CostAwareMemoryIndexConfig holds the configuration for the CostAwareMemoryIndex.
type CostAwareMemoryIndexConfig struct {
	// Size is the maximum memory size that can be used by the index.
	// Supports human-readable formats like "2GiB", "500MiB", "1GB", etc.
	Size string `json:"size,omitempty"`
}

func DefaultCostAwareMemoryIndexConfig() *CostAwareMemoryIndexConfig {
	return &CostAwareMemoryIndexConfig{
		Size: "2GiB", // 2GiB default size
	}
}

// NewCostAwareMemoryIndex creates a new CostAwareMemoryIndex instance.
func NewCostAwareMemoryIndex(cfg *CostAwareMemoryIndexConfig) (*CostAwareMemoryIndex, error) {
	if cfg == nil {
		cfg = DefaultCostAwareMemoryIndexConfig()
	}

	// Parse the size string to get byte value using go-humanize

	sizeBytes, err := humanize.ParseBytes(cfg.Size)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize cost aware index: %w", err)
	}
	cache, err := ristretto.NewCache(&ristretto.Config[string, *CostPodCache]{
		NumCounters: defaultNumCounters, // number of keys to track.
		MaxCost:     int64(sizeBytes),   // #nosec G115 , maximum cost of cache
		BufferItems: defaultBufferItems, // number of keys per Get buffer.
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize cost aware index: %w", err)
	}

	requestKeys, err := lru.New[BlockHash, BlockHash](defaultNumCounters)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize in-memory engine key map: %w", err)
	}

	return &CostAwareMemoryIndex{
		data:        cache,
		requestKeys: requestKeys,
	}, nil
}

// CostAwareMemoryIndex implements the Index interface using Ristretto cache for cost-aware memory management.
// The two caches below are kept in sync:
//   - data: requestKey -> pod cache (cost-bound by Ristretto MaxCost)
//   - requestKeys: engineKey -> requestKey (LRU to cap mapping size)
//
// Add always writes both maps; Evict removes pods and, when empty, removes
// both the requestKey entry and its engineKey mapping to avoid dangling keys.
type CostAwareMemoryIndex struct {
	// data holds the mapping of request keys to sets of pod identifiers.
	data *ristretto.Cache[string, *CostPodCache]
	// requestKeys holds the mapping of engine keys to request keys.
	requestKeys *lru.Cache[BlockHash, BlockHash]
	// mu protects concurrent access to the index operations
	mu sync.RWMutex
}

func (m *CostAwareMemoryIndex) MaxCost() int64 {
	return m.data.MaxCost()
}

// CostPodCache wraps a sync.Map of PodEntry and provides cost calculation for memory usage estimation.
type CostPodCache struct {
	cache sync.Map // map[string]*PodEntry (key: "podID@tier")
	// size tracks the number of entries in cache for O(1) Len().
	size atomic.Int64
}

// Add adds or updates a PodEntry in the cache, merging StoredGroups if the entry exists.
func (c *CostPodCache) Add(entry PodEntry) {
	cacheKey := podCacheKey(entry.PodIdentifier, entry.DeviceTier, entry.Speculative)

	// Try to load existing entry
	if existingVal, loaded := c.cache.Load(cacheKey); loaded {
		if existingEntry, ok := existingVal.(*PodEntry); ok {
			// Merge StoredGroups
			existingEntry.StoredGroups = mergeGroupsUnique(existingEntry.StoredGroups, entry.StoredGroups)
			// Store updated entry
			c.cache.Store(cacheKey, existingEntry)
		}
	} else {
		// Create new entry
		newEntry := &PodEntry{
			PodIdentifier: entry.PodIdentifier,
			DeviceTier:    entry.DeviceTier,
			Speculative:   entry.Speculative,
			StoredGroups:  mergeGroupsUnique(nil, entry.StoredGroups),
		}
		c.cache.Store(cacheKey, newEntry)
		c.size.Add(1)
	}
}

// Delete removes a PodEntry from the cache entirely.
func (c *CostPodCache) Delete(entry PodEntry) {
	cacheKey := podCacheKey(entry.PodIdentifier, entry.DeviceTier, entry.Speculative)
	if _, loaded := c.cache.LoadAndDelete(cacheKey); loaded {
		c.size.Add(-1)
	}
}

// RemoveGroups removes specified groups from a PodEntry's StoredGroups.
// If no groups remain, the entry is deleted.
func (c *CostPodCache) RemoveGroups(entry PodEntry) bool {
	cacheKey := podCacheKey(entry.PodIdentifier, entry.DeviceTier, entry.Speculative)

	existingVal, loaded := c.cache.Load(cacheKey)
	if !loaded {
		return false
	}

	existingEntry, ok := existingVal.(*PodEntry)
	if !ok {
		return false
	}

	// Remove specified groups
	updatedGroups := removeGroups(existingEntry.StoredGroups, entry.StoredGroups)

	if len(updatedGroups) == 0 {
		// No groups left, delete the entry
		c.cache.Delete(cacheKey)
		c.size.Add(-1)
		return true
	}

	// Update with remaining groups
	existingEntry.StoredGroups = updatedGroups
	c.cache.Store(cacheKey, existingEntry)
	return false
}

// Len returns the number of entries in the cache.
func (c *CostPodCache) Len() int {
	return int(c.size.Load())
}

// CalculateByteSize estimates memory usage for ristretto cost calculation.
// This is an approximation used for cache eviction decisions.
func (c *CostPodCache) CalculateByteSize(keyStr string) int64 {
	var totalBytes int64
	var entryCount int64

	// Key string memory usage
	totalBytes += int64(len(keyStr))

	// CostPodCache struct overhead (sync.Map overhead)
	totalBytes += 64 // approximate sync.Map overhead

	// Count entries and calculate their size
	c.cache.Range(func(key, value interface{}) bool {
		// key is now a string, value is *PodEntry
		keyStr, okKey := key.(string)
		entry, okEntry := value.(*PodEntry)
		if !okKey || !okEntry {
			return true
		}

		entryCount++
		totalBytes += int64(len(keyStr))                 // cache key string
		totalBytes += int64(len(entry.PodIdentifier))    // PodIdentifier string content
		totalBytes += int64(len(entry.DeviceTier))       // DeviceTier string content
		totalBytes += int64(len(entry.StoredGroups) * 8) // StoredGroups slice (8 bytes per int)
		totalBytes += 32                                 // string headers (16 bytes each for 2 strings)
		totalBytes += 24                                 // slice header for StoredGroups
		totalBytes += 8                                  // pointer to PodEntry
		totalBytes += 8                                  // struct padding/alignment
		return true
	})

	// sync.Map overhead estimation
	if entryCount > 0 {
		// Map overhead: assuming 24 bytes per entry (key+value+metadata in sync.Map)
		totalBytes += entryCount * 24
	}

	return totalBytes
}

var _ Index = &CostAwareMemoryIndex{}

// Add adds a set of keys and their associated pod entries to the index backend.
// If engineKeys is nil, only requestKey -> PodEntry mappings are created (no engineKey -> requestKey mapping).
// This is used for speculative entries where engine keys are not yet known.
func (m *CostAwareMemoryIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}
	if engineKeys != nil && len(engineKeys) != len(requestKeys) {
		return fmt.Errorf("mismatch between engine keys and request keys length")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Add")

	for i, requestKey := range requestKeys {
		// Store engineKey -> requestKey mapping (only if engineKeys provided)
		if engineKeys != nil {
			m.requestKeys.Add(engineKeys[i], requestKey)
		}

		keyStr := requestKey.String()
		podCache, found := m.data.Get(keyStr)
		if !found {
			podCache = &CostPodCache{}
		}

		for _, entry := range entries {
			podCache.Add(entry)
		}

		// Calculate the actual cost for this cache entry
		cost := podCache.CalculateByteSize(keyStr)
		m.data.Set(keyStr, podCache, cost)
		traceLogger.Info("added pods to key", "requestKey", requestKey, "pods", entries, "cost-bytes", cost)
	}
	m.data.Wait()
	return nil
}

func (m *CostAwareMemoryIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(requestKeys) == 0 {
		return nil, fmt.Errorf("no keys provided for lookup")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Lookup")

	podsPerKey := make(map[BlockHash][]PodEntry)
	highestHitIdx := 0

	for idx, key := range requestKeys {
		keyStr := key.String()
		if pods, found := m.data.Get(keyStr); found { //nolint:nestif // TODO: can this be optimized?
			if pods == nil || pods.Len() == 0 {
				traceLogger.Info("no pods found for key, cutting search", "key", key)
				return podsPerKey, nil // early stop since prefix-chain breaks here
			}

			highestHitIdx = idx

			if podIdentifierSet.Len() == 0 {
				// If no pod identifiers are provided, return all pods
				pods.cache.Range(func(k, value interface{}) bool {
					if pod, ok := value.(*PodEntry); ok {
						podsPerKey[key] = append(podsPerKey[key], *pod)
					}
					return true
				})
			} else {
				// Filter pods based on the provided pod identifiers
				pods.cache.Range(func(k, value interface{}) bool {
					if pod, ok := value.(*PodEntry); ok {
						if podIdentifierSet.Has(pod.PodIdentifier) {
							podsPerKey[key] = append(podsPerKey[key], *pod)
						}
					}
					return true
				})
			}
		} else {
			traceLogger.Info("key not found in index", "key", key)
		}
	}

	traceLogger.Info("lookup completed", "highest-hit-index", highestHitIdx,
		"pods-per-key", podsPerKeyPrintHelper(podsPerKey))

	return podsPerKey, nil
}

// Evict removes a key and its associated pod entries from the index backend.
// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
// or a RequestKey (used directly for speculative entries without engineKey mapping).
func (m *CostAwareMemoryIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("kvblock.CostAwareMemoryIndex.Evict")

	var requestKey BlockHash
	hasEngineKeyMapping := false

	switch keyType {
	case EngineKey:
		rk, found := m.requestKeys.Get(key)
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

	keyStr := requestKey.String()
	podCache, found := m.data.Get(keyStr)
	if !found || podCache == nil {
		if hasEngineKeyMapping {
			traceLogger.Info("requestKey not found in index, cleaning up engineKey", "requestKey", requestKey, "engineKey", key)
			m.requestKeys.Remove(key)
		} else {
			traceLogger.Info("key not found in index, nothing to evict", "key", key)
		}
		return nil
	}

	podCacheLenBefore := podCache.Len()

	for _, entry := range entries {
		// Remove groups from the entry; if no groups remain, the entry is deleted
		podCache.RemoveGroups(entry)
	}

	if podCache.Len() == 0 {
		m.data.Del(keyStr)
		if hasEngineKeyMapping {
			m.requestKeys.Remove(key)
		}
		traceLogger.Info("removed requestKey from index as no pods remain", "requestKey", requestKey, "key", key)
	} else if podCacheLenBefore != podCache.Len() {
		m.data.Set(keyStr, podCache, podCache.CalculateByteSize(keyStr))
		traceLogger.Info("evicted pods from key", "requestKey", requestKey, "key", key, "keyType", keyType, "pods", entries)
	}
	m.data.Wait()
	return nil
}

// GetRequestKey returns the requestKey associated with the given engineKey.
// Returns an error if the engineKey is not mapped (e.g., evicted earlier).
func (m *CostAwareMemoryIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	requestKey, found := m.requestKeys.Get(engineKey)
	if !found {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}
	return requestKey, nil
}
