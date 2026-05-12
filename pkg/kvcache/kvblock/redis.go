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

// RedisIndexConfig holds the configuration for the RedisIndex. It supports
// three connection modes driven by the (Address, Addresses, MasterName) fields,
// dispatched at runtime by go-redis's NewUniversalClient:
//
//   - Address set (standalone, single URL): "redis://host:6379" or "valkey://host:6379"
//   - Addresses set + MasterName set: Sentinel mode (MasterName names the replica set)
//   - Addresses set without MasterName: Cluster mode when len > 1, standalone when len == 1
//
// Address and Addresses are mutually exclusive; setting both is an error.
// This configuration supports both Redis and Valkey backends since they are API-compatible.
type RedisIndexConfig struct {
	// Address is the standalone URL form. Mutually exclusive with Addresses.
	Address string `json:"address,omitempty"`
	// Addresses lists one or more host:port entries. Combine with MasterName to
	// select Sentinel mode; without MasterName multiple entries become Cluster.
	Addresses []string `json:"addresses,omitempty"`
	// MasterName names the Sentinel-managed replica set; required for Sentinel mode.
	MasterName string `json:"masterName,omitempty"`
	// Username for AUTH (Redis ACL); falls back to URL-encoded credentials in Address.
	Username string `json:"username,omitempty"`
	// Password for AUTH; falls back to URL-encoded credentials in Address.
	Password string `json:"password,omitempty"`
	// DB selects the logical database (Redis only; Cluster mode requires DB 0).
	// Falls back to the DB component of Address when set to 0.
	DB int `json:"db,omitempty"`

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
// This constructor supports both Redis and Valkey backends, and routes between
// standalone, Sentinel, and Cluster modes via redis.NewUniversalClient based on
// the (Address, Addresses, MasterName) fields in RedisIndexConfig.
func NewRedisIndex(config *RedisIndexConfig) (Index, error) {
	if config == nil {
		config = DefaultRedisIndexConfig()
	}

	// Normalize the backend type
	if config.BackendType == "" {
		config.BackendType = "redis"
	}

	opts, err := universalOptionsFromRedisIndexConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to build %s options: %w", config.BackendType, err)
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

	redisClient := redis.NewUniversalClient(opts)
	if err := redisClient.Ping(context.Background()).Err(); err != nil {
		_ = redisClient.Close()
		return nil, fmt.Errorf("failed to connect to %s: %w", config.BackendType, err)
	}

	return &RedisIndex{
		RedisClient: redisClient,
		BackendType: config.BackendType,
		EnableRDMA:  config.EnableRDMA,
	}, nil
}

// universalOptionsFromRedisIndexConfig builds redis.UniversalOptions from the
// provided RedisIndexConfig. The routing rule between connection modes is:
//
//   - cfg.Address set: parsed as a redis:// (or valkey://) URL; becomes a
//     single-address standalone client. Address and Addresses are mutually
//     exclusive.
//   - cfg.Addresses set + cfg.MasterName set: Sentinel mode. Addresses are
//     treated as sentinel endpoints.
//   - cfg.Addresses set without MasterName: multi-entry becomes Cluster,
//     single-entry behaves like standalone.
//
// go-redis's NewUniversalClient performs the final dispatch based on these
// fields; we only assemble the option struct and surface the URL form's
// credentials/db settings into the Universal options.
func universalOptionsFromRedisIndexConfig(cfg *RedisIndexConfig) (*redis.UniversalOptions, error) {
	if cfg.Address != "" && len(cfg.Addresses) > 0 {
		return nil, errors.New("redis index config: `address` and `addresses` are mutually exclusive")
	}

	if cfg.Address == "" && len(cfg.Addresses) == 0 {
		return nil, errors.New("redis index config: one of `address` or `addresses` must be set")
	}

	if cfg.Address != "" {
		// Normalize valkey:// schemes to redis:// for protocol compatibility,
		// then add a default redis:// prefix for bare host:port strings.
		addr := cfg.Address
		switch {
		case strings.HasPrefix(addr, "valkey://"):
			addr = strings.Replace(addr, "valkey://", "redis://", 1)
		case strings.HasPrefix(addr, "valkeys://"):
			addr = strings.Replace(addr, "valkeys://", "rediss://", 1)
		case !strings.HasPrefix(addr, "redis://") &&
			!strings.HasPrefix(addr, "rediss://") &&
			!strings.HasPrefix(addr, "unix://"):
			addr = "redis://" + addr
		}

		parsed, err := redis.ParseURL(addr)
		if err != nil {
			return nil, fmt.Errorf("parse redis address %q: %w", cfg.Address, err)
		}

		return &redis.UniversalOptions{
			Addrs:     []string{parsed.Addr},
			Username:  firstNonEmpty(cfg.Username, parsed.Username),
			Password:  firstNonEmpty(cfg.Password, parsed.Password),
			DB:        coalesceDB(cfg.DB, parsed.DB),
			TLSConfig: parsed.TLSConfig,
		}, nil
	}

	// Sentinel / Cluster / multi-address standalone path.
	return &redis.UniversalOptions{
		Addrs:      cfg.Addresses,
		MasterName: cfg.MasterName,
		Username:   cfg.Username,
		Password:   cfg.Password,
		DB:         cfg.DB,
	}, nil
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// coalesceDB returns the explicitly configured DB value when non-zero, otherwise
// the value parsed from the URL. Zero is the default and ambiguous, so we
// prefer the URL's value only when the config field is left unset.
func coalesceDB(cfgDB, urlDB int) int {
	if cfgDB != 0 {
		return cfgDB
	}
	return urlDB
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
//
// RedisClient is a redis.UniversalClient so the same struct serves standalone,
// Sentinel, and Cluster deployments transparently. All command surfaces used
// here (HSet/HDel/HKeys, Pipeline, Eval) are identical across the three
// implementations behind the interface.
type RedisIndex struct {
	RedisClient redis.UniversalClient
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
		if engineKeys != nil {
			pipe.Set(ctx, redisEngineKey(engineKeys[i]), redisKey, 0)
		}
		for _, entry := range entries {
			// Use HSet to add the pod identifier as a field in the hash
			pipe.HSet(ctx, redisKey, entry.String(), "")
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
