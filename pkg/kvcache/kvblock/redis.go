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

	// Pre-load Lua scripts so EvalSha calls in pipelines never get NOSCRIPT errors.
	if err := clearPodEntryScript.Load(context.Background(), redisClient).Err(); err != nil {
		return nil, fmt.Errorf("failed to load Lua scripts on %s: %w", config.BackendType, err)
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

// pruneRequestKeyScript atomically deletes a request key hash if it contains no pods.
var pruneRequestKeyScript = redis.NewScript(`
	local hashLen = redis.call('HLEN', KEYS[1])
	if hashLen == 0 then
		redis.call('DEL', KEYS[1])
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
// When engineKeys is non-nil, the mapping type is inferred from the ratio of array lengths.
func (r *RedisIndex) Add(ctx context.Context, engineKeys, requestKeys []BlockHash, entries []PodEntry) error {
	if len(requestKeys) == 0 || len(entries) == 0 {
		return fmt.Errorf("no keys or entries provided for adding to index")
	}

	pipe := r.RedisClient.Pipeline()

	// Build engine->request mappings when engine keys are provided.
	// The ratio of array lengths determines the mapping type:
	//   equal  (4 eng, 4 req) -> 1:1   E0->R0, E1->R1, ...
	//   many:1 (4 eng, 1 req) -> E0->R0, E1->R0, E2->R0, E3->R0
	//   1:many (1 eng, 4 req) -> E0->[R0, R1, R2, R3]
	//
	// Also build the inverse (requestKey -> []engineKey strings) for the reverse index.
	// In the many:1 case multiple engine keys map to the same request key; all are recorded.
	requestToEngineKeys := make(map[string][]string, len(requestKeys))
	if engineKeys != nil {
		n := max(len(engineKeys), len(requestKeys))
		for i := 0; i < n; i++ {
			ek := engineKeys[i*len(engineKeys)/n]
			rk := requestKeys[i*len(requestKeys)/n]
			pipe.ZAdd(ctx, redisEngineKey(ek), redis.Z{Score: float64(i), Member: rk.String()})
			requestToEngineKeys[rk.String()] = append(requestToEngineKeys[rk.String()], ek.String())
		}
	}

	// Store requestKey -> PodEntry mappings for all request keys.
	for _, requestKey := range requestKeys {
		redisKey := requestKey.String()
		// Join all engine keys for this request key; may be empty for speculative entries.
		engineKeyStr := strings.Join(requestToEngineKeys[redisKey], ",")
		for _, entry := range entries {
			pipe.HSet(ctx, redisKey, entry.String(), "")
			// Store reverse-index: pod:<podIdentifier> hash
			//   field = entry.String()  (e.g. "10.0.0.1:8080@gpu")
			//   value = "<requestKey>:<ek1,ek2,...>"  (engine keys may be empty)
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

	switch keyType {
	case EngineKey:
		rks, err := r.getRequestKeys(ctx, key)
		if err != nil || len(rks) == 0 {
			// Engine key not found in mapping — nothing to evict
			return nil //nolint:nilerr // intentional: missing engine key means nothing to evict
		}
		for _, rk := range rks {
			if err := r.evictPodsFromRequestKey(ctx, rk, entries); err != nil {
				return err
			}
		}
		// Clean up the engine key set
		if err := r.RedisClient.Del(ctx, redisEngineKey(key)).Err(); err != nil {
			return fmt.Errorf("failed to delete engine key mapping: %w", err)
		}
		return nil
	case RequestKey:
		return r.evictPodsFromRequestKey(ctx, key, entries)
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictPodsFromRequestKey removes the given pod entries from a single request key.
// If the pod hash becomes empty, the request key is removed and any engine key
// sorted sets that still reference it are cleaned up via ZREM.
func (r *RedisIndex) evictPodsFromRequestKey(ctx context.Context, requestKey BlockHash, entries []PodEntry) error {
	redisKey := requestKey.String()

	// Collect engine keys associated with this request key from the pod reverse-index
	// before deleting entries — needed to ZREM the request key from their sorted sets
	// in the many:1 case (multiple engine keys → same request key).
	engineKeySet := make(map[string]struct{})
	for _, entry := range entries {
		val, err := r.RedisClient.HGet(ctx, podIdentifierKey(entry.PodIdentifier), entry.String()).Result()
		if err != nil {
			continue // not found or Redis error; skip
		}
		_, engineKeysStr, _ := strings.Cut(val, ":")
		for _, ekStr := range strings.Split(engineKeysStr, ",") {
			if ekStr != "" {
				engineKeySet[redisEngineKey(BlockHash(mustParseUint64(ekStr)))] = struct{}{}
			}
		}
	}

	pipe := r.RedisClient.Pipeline()
	for _, entry := range entries {
		pipe.HDel(ctx, redisKey, entry.String())
		pipe.HDel(ctx, podIdentifierKey(entry.PodIdentifier), entry.String())
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to evict entries from Redis: %w", err)
	}

	// Atomically delete the request key hash if it's now empty.
	pruned, err := pruneRequestKeyScript.Run(ctx, r.RedisClient, []string{redisKey}).Int()
	if err != nil {
		return fmt.Errorf("failed to prune empty request key: %w", err)
	}

	// If the request key was pruned, remove it from every engine key sorted set
	// and delete the sorted set itself when it becomes empty.
	if pruned == 1 && len(engineKeySet) > 0 {
		pipe = r.RedisClient.Pipeline()
		for ekRedisKey := range engineKeySet {
			pipe.ZRem(ctx, ekRedisKey, redisKey)
		}
		if _, err := pipe.Exec(ctx); err != nil {
			return fmt.Errorf("failed to clean up engine key sorted sets: %w", err)
		}
		// Delete sorted sets that became empty after the ZREM.
		for ekRedisKey := range engineKeySet {
			card, err := r.RedisClient.ZCard(ctx, ekRedisKey).Result()
			if err == nil && card == 0 {
				r.RedisClient.Del(ctx, ekRedisKey) //nolint:errcheck // best-effort cleanup
			}
		}
	}

	return nil
}

// getRequestKeys returns all request keys mapped to the given engine key.
func (r *RedisIndex) getRequestKeys(ctx context.Context, engineKey BlockHash) ([]BlockHash, error) {
	vals, err := r.RedisClient.ZRange(ctx, redisEngineKey(engineKey), 0, -1).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return nil, nil
		}
		return nil, err
	}

	rks := make([]BlockHash, 0, len(vals))
	for _, val := range vals {
		hash, err := strconv.ParseUint(val, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("invalid hash format: %s", val)
		}
		rks = append(rks, BlockHash(hash))
	}
	return rks, nil
}

// GetRequestKey returns the last request key (highest score) associated with the given engineKey.
func (r *RedisIndex) GetRequestKey(ctx context.Context, engineKey BlockHash) (BlockHash, error) {
	vals, err := r.RedisClient.ZRevRange(ctx, redisEngineKey(engineKey), 0, 0).Result()
	if err != nil {
		return EmptyBlockHash, err
	}
	if len(vals) == 0 {
		return EmptyBlockHash, fmt.Errorf("engine key not found: %s", engineKey.String())
	}

	hash, err := strconv.ParseUint(vals[0], 10, 64)
	if err != nil {
		return EmptyBlockHash, fmt.Errorf("invalid hash format: %s", vals[0])
	}
	return BlockHash(hash), nil
}

func redisEngineKey(engineKey BlockHash) string {
	if engineKey == EmptyBlockHash {
		return ""
	}
	return "engine:" + engineKey.String()
}

func podIdentifierKey(podIdentifier string) string {
	return "pod:" + podIdentifier
}

// clearPodEntryScript atomically removes a single pod-entry field from the
// request-key hash AND from the pod reverse-index hash, then prunes each
// engine-key sorted set and the request-key hash when they become empty.
//
// KEYS[1] = request-key hash      (e.g. "10633516")
// KEYS[2] = pod reverse-index hash (e.g. "pod:10.0.0.1:8080")
// ARGV[1] = pod entry field        (e.g. "10.0.0.1:8080@gpu")
// ARGV[2..N] = engine-key redis keys to ZREM from (optional, one per engine key)
var clearPodEntryScript = redis.NewScript(`
	redis.call('HDEL', KEYS[1], ARGV[1])
	redis.call('HDEL', KEYS[2], ARGV[1])
	if redis.call('HLEN', KEYS[2]) == 0 then
		redis.call('DEL', KEYS[2])
	end
	if redis.call('HLEN', KEYS[1]) == 0 then
		redis.call('DEL', KEYS[1])
		for i = 2, #ARGV do
			redis.call('ZREM', ARGV[i], KEYS[1])
			if redis.call('ZCARD', ARGV[i]) == 0 then
				redis.call('DEL', ARGV[i])
			end
		end
	end
	return 1
`)

// Clear removes all index entries for the given podEntry.
//
// The pod reverse-index hash (pod:<podIdentifier>) stores:
//
//	field = entry.String()            e.g. "10.0.0.1:8080@gpu"
//	value = "<requestKey>:<ek1,ek2>" (engine keys comma-separated; may be empty for speculative entries)
func (r *RedisIndex) Clear(ctx context.Context, podEntry PodEntry) error {
	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.Clear")

	podKey := podIdentifierKey(podEntry.PodIdentifier)

	// HGETALL returns all {entryString -> "requestKey:<ek1,ek2,...>"} pairs in one RTT.
	fields, err := r.RedisClient.HGetAll(ctx, podKey).Result()
	if err != nil {
		return fmt.Errorf("failed to get pod reverse-index for %s: %w", podEntry.PodIdentifier, err)
	}
	if len(fields) == 0 {
		logger.Info("pod not found in reverse index, nothing to clear", "podEntry", podEntry)
		return nil
	}

	pipe := r.RedisClient.Pipeline()
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

		// meta = "<requestKey>:<ek1,ek2,...>"  (engine keys part may be empty)
		requestKeyStr, engineKeysStr, _ := strings.Cut(meta, ":")

		// Build ARGV: [podEntryField, engineRedisKey1, engineRedisKey2, ...]
		argv := []interface{}{entryStr}
		if engineKeysStr != "" {
			for _, ekStr := range strings.Split(engineKeysStr, ",") {
				if ekStr != "" {
					argv = append(argv, redisEngineKey(BlockHash(mustParseUint64(ekStr))))
				}
			}
		}

		pipe.EvalSha(ctx, clearPodEntryScript.Hash(), []string{requestKeyStr, podKey}, argv...)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to clear pod entries from Redis index: %w", err)
	}

	logger.Info("Cleared pod entries from Redis index", "podEntry", podEntry)
	return nil
}

// mustParseUint64 parses a uint64 string, returning 0 on failure.
func mustParseUint64(s string) uint64 {
	v, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return 0
	}
	return v
}
