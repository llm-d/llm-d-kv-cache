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
	results := make([]*redis.StringSliceCmd, len(requestKeys))

	// queue an HKeys command for each key in the pipeline
	for i, key := range requestKeys {
		// HKeys gets all field names
		results[i] = pipe.HKeys(ctx, key.String())
	}

	_, execErr := pipe.Exec(ctx)
	if execErr != nil {
		return nil, fmt.Errorf("redis pipeline execution failed: %w", execErr)
	}

	filterPods := len(podIdentifierSet) > 0 // predicate for filtering

	for idx, cmd := range results {
		key := requestKeys[idx]

		// cmd.Result() returns the slice of strings (pod IDs) which is the first layer in the mapping
		pods, cmdErr := cmd.Result()
		if cmdErr != nil {
			if !errors.Is(cmdErr, redis.Nil) {
				logger.Error(cmdErr, "failed to get pods for key", "key", key)
			}

			return podsPerKey, nil // early stop since prefix-chain breaks here
		}

		var filteredPods []PodEntry
		for _, p := range pods {
			ip := strings.SplitN(p, "@", 2)[0]
			if !filterPods || podIdentifierSet.Has(ip) {
				tier := strings.SplitN(p, "@", 2)[1]
				speculative := false
				// Strip annotation suffix e.g. "gpu[speculative]" -> "gpu"
				if idx := strings.Index(tier, "["); idx != -1 {
					speculative = strings.Contains(tier[idx:], "speculative")
					tier = tier[:idx]
				}
				filteredPods = append(filteredPods, PodEntry{PodIdentifier: ip, DeviceTier: tier, Speculative: speculative})
			}
		}

		if len(filteredPods) == 0 {
			logger.Info("no pods found for key, cutting search", "key", key)
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

	pipe := r.RedisClient.Pipeline()
	for i, requestKey := range requestKeys {
		redisKey := requestKey.String()

		// Store engineKey -> requestKey mapping (only if engineKeys provided)
		engineKeyStr := ""
		if engineKeys != nil {
			pipe.Set(ctx, redisEngineKey(engineKeys[i]), redisKey, 0)
			engineKeyStr = engineKeys[i].String()
		}
		for _, entry := range entries {
			// Use HSet to add the pod identifier as a field in the hash
			pipe.HSet(ctx, redisKey, entry.String(), "")
			// Store reverse-index: pod:<podIdentifier> hash
			//   field = entry.String()  (e.g. "10.0.0.1:8080@gpu")
			//   value = "<requestKey>:<engineKey>"  (engineKey may be empty)
			pipe.HSet(ctx, podIdentifierKey(entry.PodIdentifier), entry.String(), redisKey+":"+engineKeyStr)
		}
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to add entries to Redis: %w", err)
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
	pipe := r.RedisClient.Pipeline()

	for _, entry := range entries {
		// Use HDel to remove the pod identifier field from the hash
		pipe.HDel(ctx, redisKey, entry.String())
		// Remove the corresponding field from the pod reverse-index hash.
		pipe.HDel(ctx, podIdentifierKey(entry.PodIdentifier), entry.String())
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to evict entries from Redis: %w", err)
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

func podIdentifierKey(podIdentifier string) string {
	return "pod:" + podIdentifier
}

// clearPodEntryScript atomically removes a single pod-entry field from the
// request-key hash AND from the pod reverse-index hash, then prunes the
// engine-key string and request-key hash when they become empty.
//
// KEYS[1] = request-key hash  (e.g. "10633516")
// KEYS[2] = engine-key string (e.g. "engine:55269488")  — may be "" to skip
// KEYS[3] = pod reverse-index hash (e.g. "pod:10.0.0.1:8080")
// ARGV[1] = pod entry field  (e.g. "10.0.0.1:8080@gpu")
var clearPodEntryScript = redis.NewScript(`
	redis.call('HDEL', KEYS[1], ARGV[1])
	redis.call('HDEL', KEYS[3], ARGV[1])
	if redis.call('HLEN', KEYS[3]) == 0 then
		redis.call('DEL', KEYS[3])
	end
	if KEYS[2] ~= '' and redis.call('HLEN', KEYS[1]) == 0 then
		redis.call('DEL', KEYS[2])
		redis.call('DEL', KEYS[1])
	end
	return 1
`)

// Clear removes all index entries for the given podEntry.
//
// The pod reverse-index hash (pod:<podIdentifier>) stores:
//
//	field = entry.String()  e.g. "10.0.0.1:8080@gpu"
//	value = "<requestKey>:<engineKey>"  (engineKey may be empty for speculative entries)
func (r *RedisIndex) Clear(ctx context.Context, podEntry PodEntry) error {
	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.Clear")

	podKey := podIdentifierKey(podEntry.PodIdentifier)

	// HGETALL returns all {entryString -> "requestKey:engineKey"} pairs in one RTT.
	fields, err := r.RedisClient.HGetAll(ctx, podKey).Result()
	if err != nil {
		return fmt.Errorf("failed to get pod reverse-index for %s: %w", podEntry.PodIdentifier, err)
	}
	if len(fields) == 0 {
		logger.Info("pod not found in reverse index, nothing to clear", "podEntry", podEntry)
		return nil
	}

	for entryStr, meta := range fields {
		// Filter by DeviceTier when specified.
		// entryStr format: "<podIdentifier>@<tier>[speculative]"
		if podEntry.DeviceTier != "" {
			parts := strings.SplitN(entryStr, "@", 2)
			if len(parts) < 2 {
				continue
			}
			tier := parts[1]
			// Strip optional [speculative] suffix before comparing
			if idx := strings.Index(tier, "["); idx != -1 {
				tier = tier[:idx]
			}
			if tier != podEntry.DeviceTier {
				continue
			}
		}

		// meta = "<requestKey>:<engineKey>"  (engineKey part may be empty)
		sep := strings.LastIndex(meta, ":")
		requestKeyStr := meta
		engineKeyRedisKey := ""
		if sep >= 0 {
			requestKeyStr = meta[:sep]
			if ek := meta[sep+1:]; ek != "" {
				engineKeyRedisKey = redisEngineKey(BlockHash(mustParseUint64(ek)))
			}
		}

		if err := clearPodEntryScript.Run(
			ctx, r.RedisClient,
			[]string{requestKeyStr, engineKeyRedisKey, podKey},
			entryStr,
		).Err(); err != nil && !errors.Is(err, redis.Nil) {
			return fmt.Errorf("failed to clear pod entry %s from hash %s: %w", entryStr, requestKeyStr, err)
		}
	}

	logger.Info("Cleared pod entries from Redis index", "podEntry", podEntry)
	return nil
}

// mustParseUint64 parses a uint64 string, returning 0 on failure.
func mustParseUint64(s string) uint64 {
	v, _ := strconv.ParseUint(s, 10, 64)
	return v
}
