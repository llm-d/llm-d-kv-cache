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
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/redis/go-redis/v9"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// RedisIndexConfig holds the configuration for the RedisIndex.
// This configuration supports both Redis and Valkey backends since they are API-compatible.
type RedisIndexConfig struct {
	Address string `json:"address,omitempty"` // Redis/Valkey server address
	// BackendType specifies whether to connect to "redis" or "valkey" (optional, defaults to "redis")
	// This is mainly for documentation and future extensibility (e.g., RDMA support)
	BackendType string `json:"backendType,omitempty"`
	// EnableRDMA enables RDMA transport for Valkey when supported (experimental)
	EnableRDMA bool `json:"enableRDMA,omitempty"`
}

func DefaultRedisIndexConfig() *RedisIndexConfig {
	return &RedisIndexConfig{
		Address:     "redis://127.0.0.1:6379",
		BackendType: "redis",
		EnableRDMA:  false,
	}
}

// DefaultValkeyIndexConfig returns a default configuration for Valkey.
func DefaultValkeyIndexConfig() *RedisIndexConfig {
	return &RedisIndexConfig{
		Address:     "valkey://127.0.0.1:6379",
		BackendType: "valkey",
		EnableRDMA:  false,
	}
}

// NewRedisIndex creates a new RedisIndex instance.
// This constructor supports both Redis and Valkey backends.
func NewRedisIndex(config *RedisIndexConfig) (Index, error) {
	if config == nil {
		config = DefaultRedisIndexConfig()
	}

	// Normalize the backend type
	if config.BackendType == "" {
		config.BackendType = "redis"
	}

	// Handle address prefixing for both Redis and Valkey
	needsPrefix := !strings.HasPrefix(config.Address, "redis://") &&
		!strings.HasPrefix(config.Address, "rediss://") &&
		!strings.HasPrefix(config.Address, "valkey://") &&
		!strings.HasPrefix(config.Address, "valkeys://") &&
		!strings.HasPrefix(config.Address, "unix://")

	switch {
	case needsPrefix:
		// Default to redis:// prefix for backward compatibility
		// Valkey is API-compatible with Redis protocol
		config.Address = "redis://" + config.Address
	case strings.HasPrefix(config.Address, "valkey://"):
		// Convert valkey:// to redis:// for protocol compatibility
		config.Address = strings.Replace(config.Address, "valkey://", "redis://", 1)
	case strings.HasPrefix(config.Address, "valkeys://"):
		// Convert valkeys:// to rediss:// for SSL protocol compatibility
		config.Address = strings.Replace(config.Address, "valkeys://", "rediss://", 1)
	}

	redisOpt, err := redis.ParseURL(config.Address)
	if err != nil {
		return nil, fmt.Errorf("failed to parse %s URL: %w", config.BackendType, err)
	}

	// Future: Add RDMA configuration for Valkey when supported
	if config.BackendType == "valkey" && config.EnableRDMA {
		// TODO: Implement RDMA configuration when Valkey Go client supports it
		//
		// Note: RDMA will work if configured directly in the Valkey server instance,
		// but the Go client doesn't yet have configuration options to enable RDMA.
		// This configuration flag is a placeholder for future Go client RDMA support.
		// The connection will work with standard TCP for now.

		// Log that RDMA is requested but not yet supported in Go client
		fmt.Printf("RDMA requested for Valkey but not yet supported in Go client - using TCP\n")
	}

	redisClient := redis.NewClient(redisOpt)
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to %s: %w", config.BackendType, err)
	}

	return &RedisIndex{
		RedisClient: redisClient,
		BackendType: config.BackendType,
		EnableRDMA:  config.EnableRDMA,
	}, nil
}

// NewValkeyIndex creates a new RedisIndex instance configured for Valkey.
// This is a convenience constructor that sets up Valkey-specific defaults.
func NewValkeyIndex(config *RedisIndexConfig) (Index, error) {
	if config == nil {
		config = DefaultValkeyIndexConfig()
	} else {
		// Ensure BackendType is set to valkey
		config.BackendType = "valkey"
	}

	return NewRedisIndex(config)
}

// RedisIndex implements the Index interface
// using Redis or Valkey as the backend for KV block indexing.
type RedisIndex struct {
	RedisClient *redis.Client
	// BackendType indicates whether this is connecting to "redis" or "valkey"
	BackendType string
	// EnableRDMA indicates if RDMA transport is enabled (for Valkey)
	EnableRDMA bool
}

var _ Index = &RedisIndex{}

// pruneEngineKeyScript atomically verifies that a request key contains no pods, deleting the corresponding engine key if true.
var pruneEngineKeyScript = redis.NewScript(`
	local hashLen = redis.call('HLEN', KEYS[1])
	if hashLen == 0 then
		redis.call('DEL', KEYS[2])
		return 1
	end
	return 0
`)

// Lookup receives a list of keys and a set of pod identifiers,
// and retrieves the filtered pods associated with those keys.
// The filtering is done based on the pod identifiers provided.
// If the podIdentifierSet is empty, all pods are returned.
//
// It returns:
// 1. A map where the keys are those in (1) and the values are pod-identifiers.
// 2. An error if any occurred during the operation.
func (r *RedisIndex) Lookup(ctx context.Context, requestKeys []BlockHash,
	podIdentifierSet sets.Set[string],
) (map[BlockHash][]PodEntry, error) {
	if len(requestKeys) == 0 {
		return make(map[BlockHash][]PodEntry), nil
	}

	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.Lookup")
	podsPerKey := make(map[BlockHash][]PodEntry)

	// pipeline for single RTT
	pipe := r.RedisClient.Pipeline()
	results := make([]*redis.MapStringStringCmd, len(requestKeys))

	// queue an HGetAll command for each key in the pipeline to get all field:value pairs
	for i, key := range requestKeys {
		results[i] = pipe.HGetAll(ctx, key.String())
	}

	_, execErr := pipe.Exec(ctx)
	if execErr != nil {
		return nil, fmt.Errorf("redis pipeline execution failed: %w", execErr)
	}

	filterPods := len(podIdentifierSet) > 0 // predicate for filtering

	for idx, cmd := range results {
		key := requestKeys[idx]

		// cmd.Result() returns a map[string]string of entryKey -> JSON data
		entryMap, cmdErr := cmd.Result()
		if cmdErr != nil {
			if !errors.Is(cmdErr, redis.Nil) {
				logger.Error(cmdErr, "failed to get pods for key", "key", key)
			}

			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		if len(entryMap) == 0 {
			logger.Info("no pods found for key, cutting search", "key", key)
			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		var filteredPods []PodEntry
		for _, jsonData := range entryMap {
			var entry PodEntry
			if err := json.Unmarshal([]byte(jsonData), &entry); err != nil {
				logger.Error(err, "failed to unmarshal pod entry", "key", key, "data", jsonData)
				continue
			}

			if !filterPods || podIdentifierSet.Has(entry.PodIdentifier) {
				filteredPods = append(filteredPods, entry)
			}
		}

		if len(filteredPods) == 0 {
			logger.Info("no pods found for key after filtering, cutting search", "key", key)
			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		podsPerKey[key] = filteredPods
	}

	return podsPerKey, nil
}

// Add adds a set of keys and their associated pod entries to the index backend.
// If engineKeys is nil, only requestKey -> PodEntry mappings are created (no engineKey -> requestKey mapping).
// This is used for speculative entries where engine keys are not yet known.
func (r *RedisIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}
	if engineKeys != nil && len(engineKeys) != len(requestKeys) {
		return fmt.Errorf("mismatch between engine keys and request keys length")
	}

	for i, requestKey := range requestKeys {
		redisKey := requestKey.String()

		// Store engineKey -> requestKey mapping (only if engineKeys provided)
		if engineKeys != nil {
			if err := r.RedisClient.Set(ctx, redisEngineKey(engineKeys[i]), redisKey, 0).Err(); err != nil {
				return fmt.Errorf("failed to set engine key mapping: %w", err)
			}
		}

		for _, entry := range entries {
			entryKey := podEntryKey(entry.PodIdentifier, entry.DeviceTier)

			// Get existing entry if it exists
			existingData, err := r.RedisClient.HGet(ctx, redisKey, entryKey).Result()
			if err == nil {
				// Entry exists, merge groups
				var existingEntry PodEntry
				if err := json.Unmarshal([]byte(existingData), &existingEntry); err != nil {
					return fmt.Errorf("failed to unmarshal existing entry: %w", err)
				}

				// Merge StoredGroups
				existingEntry.StoredGroups = mergeGroupsUnique(existingEntry.StoredGroups, entry.StoredGroups)
				// Update speculative flag if new entry is confirmed
				if !entry.Speculative {
					existingEntry.Speculative = false
				}

				// Serialize and store updated entry
				data, err := json.Marshal(existingEntry)
				if err != nil {
					return fmt.Errorf("failed to marshal updated entry: %w", err)
				}
				if err := r.RedisClient.HSet(ctx, redisKey, entryKey, data).Err(); err != nil {
					return fmt.Errorf("failed to update entry in Redis: %w", err)
				}
			} else if errors.Is(err, redis.Nil) {
				// Entry doesn't exist, create new with deduplicated groups
				newEntry := PodEntry{
					PodIdentifier: entry.PodIdentifier,
					DeviceTier:    entry.DeviceTier,
					Speculative:   entry.Speculative,
					StoredGroups:  mergeGroupsUnique(nil, entry.StoredGroups),
				}

				data, err := json.Marshal(newEntry)
				if err != nil {
					return fmt.Errorf("failed to marshal new entry: %w", err)
				}
				if err := r.RedisClient.HSet(ctx, redisKey, entryKey, data).Err(); err != nil {
					return fmt.Errorf("failed to add entry to Redis: %w", err)
				}
			} else {
				return fmt.Errorf("failed to check existing entry: %w", err)
			}
		}
	}

	return nil
}

// Evict removes a key and its associated pod entries from the index backend.
// keyType indicates whether the key is an EngineKey (requires engine→request lookup)
// or a RequestKey (used directly for speculative entries without engineKey mapping).
func (r *RedisIndex) Evict(ctx context.Context, key BlockHash, keyType KeyType, entries []PodEntry) error {
	if len(entries) == 0 {
		return fmt.Errorf("no entries provided for eviction from index")
	}

	var requestKey BlockHash
	hasEngineKeyMapping := false

	switch keyType {
	case EngineKey:
		rk, err := r.GetRequestKey(ctx, key)
		if err != nil {
			// Engine key not found in mapping — nothing to evict
			return nil //nolint:nilerr // intentional: missing engine key means nothing to evict
		}
		requestKey = rk
		hasEngineKeyMapping = true
	case RequestKey:
		requestKey = key
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}

	redisKey := requestKey.String()

	for _, entry := range entries {
		entryKey := podEntryKey(entry.PodIdentifier, entry.DeviceTier)

		// Get existing entry
		existingData, err := r.RedisClient.HGet(ctx, redisKey, entryKey).Result()
		if errors.Is(err, redis.Nil) {
			// Entry doesn't exist, nothing to evict
			continue
		} else if err != nil {
			return fmt.Errorf("failed to get existing entry: %w", err)
		}

		// Deserialize existing entry
		var existingEntry PodEntry
		if err := json.Unmarshal([]byte(existingData), &existingEntry); err != nil {
			return fmt.Errorf("failed to unmarshal existing entry: %w", err)
		}

		// Remove specified groups
		updatedGroups := removeGroups(existingEntry.StoredGroups, entry.StoredGroups)

		if len(updatedGroups) == 0 {
			// No groups left, remove the entire entry
			if err := r.RedisClient.HDel(ctx, redisKey, entryKey).Err(); err != nil {
				return fmt.Errorf("failed to delete entry from Redis: %w", err)
			}
		} else {
			// Update with remaining groups
			existingEntry.StoredGroups = updatedGroups
			data, err := json.Marshal(existingEntry)
			if err != nil {
				return fmt.Errorf("failed to marshal updated entry: %w", err)
			}
			if err := r.RedisClient.HSet(ctx, redisKey, entryKey, data).Err(); err != nil {
				return fmt.Errorf("failed to update entry in Redis: %w", err)
			}
		}
	}

	// Atomically check hash length and delete engine key if empty (only if engine key mapping exists)
	if hasEngineKeyMapping {
		if err := pruneEngineKeyScript.Run(ctx, r.RedisClient, []string{redisKey, redisEngineKey(key)}).Err(); err != nil {
			return fmt.Errorf("failed to check hash length and cleanup engine key: %w", err)
		}
	}

	return nil
}

func (r *RedisIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	val, err := r.RedisClient.Get(ctx, redisEngineKey(engineKey)).Result()
	if err != nil {
		return EmptyBlockHash, err
	}

	hash, err := strconv.ParseUint(val, 10, 64)
	if err != nil {
		return EmptyBlockHash, fmt.Errorf("invalid hash format: %s", val)
	}

	return BlockHash(hash), nil
}

func redisEngineKey(engineKey BlockHash) string {
	return "engine:" + engineKey.String()
}

// podEntryKey generates a hash field key for a pod entry.
// Format: "podIdentifier@deviceTier"
func podEntryKey(podIdentifier, deviceTier string) string {
	return podIdentifier + "@" + deviceTier
}
