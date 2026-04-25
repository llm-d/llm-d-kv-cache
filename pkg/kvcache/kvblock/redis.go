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
	"time"

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
	// gen tracks per-pod generation counters for O(1) Clear via lazy invalidation.
	// Cached locally; bumped on Clear (and would be replicated cross-process via
	// a Redis INCR + pubsub in a multi-replica deployment — out of scope here).
	gen podGenTracker
	// sweepCh is signalled by Clear when a background sweeper is running.
	sweepCh chan struct{}
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

// pruneEngineKeyScript atomically deletes an engine key mapping only if all
// associated request key hashes are empty. This prevents a TOCTOU race where a
// concurrent Add could insert into a request key between checking and deleting.
// KEYS[1] = engine key ("engine:<hash>"), KEYS[2..N] = request key hashes.
var pruneEngineKeyScript = redis.NewScript(`
	for i = 2, #KEYS do
		if redis.call('HLEN', KEYS[i]) > 0 then
			return 0
		end
	end
	redis.call('DEL', KEYS[1])
	return 1
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

	filterPods := len(podIdentifierSet) > 0
	needGenFilter := r.gen.anyClears() != 0
	var gens *genCache
	if needGenFilter {
		gens = r.gen.snapshot()
	}

	// pipeline for single RTT.
	// Fast path: if no Clear has ever happened, we only need field names (HKeys),
	// which avoids fetching and decoding the per-entry generation values.
	pipe := r.RedisClient.Pipeline()

	if !needGenFilter {
		results := make([]*redis.StringSliceCmd, len(requestKeys))
		for i, key := range requestKeys {
			results[i] = pipe.HKeys(ctx, key.String())
		}
		if _, err := pipe.Exec(ctx); err != nil {
			return nil, fmt.Errorf("redis pipeline execution failed: %w", err)
		}
		for idx, cmd := range results {
			key := requestKeys[idx]
			pods, cmdErr := cmd.Result()
			if cmdErr != nil {
				if !errors.Is(cmdErr, redis.Nil) {
					logger.Error(cmdErr, "failed to get pods for key", "key", key)
				}
				return podsPerKey, nil
			}
			var filteredPods []PodEntry
			for _, p := range pods {
				entry, ok := parseRedisPodField(p)
				if !ok {
					continue
				}
				if filterPods && !podIdentifierSet.Has(entry.PodIdentifier) {
					continue
				}
				filteredPods = append(filteredPods, entry)
			}
			if len(filteredPods) == 0 {
				logger.Info("no pods found for key, cutting search", "key", key)
				return podsPerKey, nil
			}
			podsPerKey[key] = filteredPods
		}
		return podsPerKey, nil
	}

	// Slow path: at least one Clear has happened, so we need values to filter by gen.
	results := make([]*redis.MapStringStringCmd, len(requestKeys))
	for i, key := range requestKeys {
		results[i] = pipe.HGetAll(ctx, key.String())
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("redis pipeline execution failed: %w", err)
	}

	for idx, cmd := range results {
		key := requestKeys[idx]
		pods, cmdErr := cmd.Result()
		if cmdErr != nil {
			if !errors.Is(cmdErr, redis.Nil) {
				logger.Error(cmdErr, "failed to get pods for key", "key", key)
			}
			return podsPerKey, nil
		}
		var filteredPods []PodEntry
		for p, genStr := range pods {
			entry, ok := parseRedisPodField(p)
			if !ok {
				continue
			}
			if filterPods && !podIdentifierSet.Has(entry.PodIdentifier) {
				continue
			}
			stampedGen, _ := strconv.ParseUint(genStr, 10, 64)
			if stampedGen < gens.current(entry.PodIdentifier) {
				continue
			}
			filteredPods = append(filteredPods, entry)
		}
		if len(filteredPods) == 0 {
			logger.Info("no pods found for key, cutting search", "key", key)
			return podsPerKey, nil
		}
		podsPerKey[key] = filteredPods
	}

	return podsPerKey, nil
}

// parseRedisPodField parses a Redis hash field of the form
// "<podIdentifier>@<tier>[speculative]" back into a PodEntry. Returns false
// if the field doesn't contain the expected separator.
func parseRedisPodField(p string) (PodEntry, bool) {
	at := strings.IndexByte(p, '@')
	if at < 0 {
		return PodEntry{}, false
	}
	id := p[:at]
	tier := p[at+1:]
	speculative := false
	if i := strings.IndexByte(tier, '['); i != -1 {
		speculative = strings.Contains(tier[i:], "speculative")
		tier = tier[:i]
	}
	return PodEntry{PodIdentifier: id, DeviceTier: tier, Speculative: speculative}, true
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
	if engineKeys != nil {
		n := max(len(engineKeys), len(requestKeys))
		for i := 0; i < n; i++ {
			ek := engineKeys[i*len(engineKeys)/n]
			rk := requestKeys[i*len(requestKeys)/n]
			pipe.ZAdd(ctx, redisEngineKey(ek), redis.Z{Score: float64(i), Member: rk.String()})
		}
	}

	// Store requestKey -> PodEntry mappings for all request keys.
	// The HSet value is the entry's admission generation (per-pod), so Lookup can
	// filter entries that pre-date the latest Clear without an extra round-trip.
	for _, requestKey := range requestKeys {
		redisKey := requestKey.String()
		for _, entry := range entries {
			genStr := strconv.FormatUint(r.gen.current(entry.PodIdentifier), 10)
			pipe.HSet(ctx, redisKey, entry.String(), genStr)
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
		keys := make([]string, 0, 1+len(rks))
		keys = append(keys, redisEngineKey(key))
		for _, rk := range rks {
			keys = append(keys, rk.String())
		}
		if err := pruneEngineKeyScript.Run(ctx, r.RedisClient, keys).Err(); err != nil && !errors.Is(err, redis.Nil) {
			return fmt.Errorf("failed to prune engine key mapping: %w", err)
		}
		return nil
	case RequestKey:
		return r.evictPodsFromRequestKey(ctx, key, entries)
	default:
		return fmt.Errorf("unknown key type: %d", keyType)
	}
}

// evictPodsFromRequestKey removes the given pod entries from a single request key.
// If the pod hash becomes empty, the request key is removed.
func (r *RedisIndex) evictPodsFromRequestKey(ctx context.Context, requestKey BlockHash, entries []PodEntry) error {
	redisKey := requestKey.String()
	pipe := r.RedisClient.Pipeline()

	for _, entry := range entries {
		pipe.HDel(ctx, redisKey, entry.String())
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to evict entries from Redis: %w", err)
	}

	// Atomically delete the request key hash if it's now empty
	if err := pruneRequestKeyScript.Run(ctx, r.RedisClient, []string{redisKey}).Err(); err != nil {
		return fmt.Errorf("failed to prune empty request key: %w", err)
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
	return "engine:" + engineKey.String()
}

// Clear invalidates all entries for the given podEntry by bumping the pod's
// generation counter. Stale entries are filtered at Lookup time and reclaimed
// lazily (or eagerly via Sweep). O(1) — a single atomic increment locally.
//
// NOTE: in a multi-replica deployment, this would also issue a Redis INCR on
// `podgen:<id>` and propagate via pubsub so other replicas converge. That cross-
// process plumbing is intentionally elided in this prototype to keep the
// benchmark apples-to-apples with the in-process backends.
func (r *RedisIndex) Clear(ctx context.Context, podEntry PodEntry) error {
	r.gen.bump(podEntry.PodIdentifier)
	log.FromContext(ctx).WithName("kvblock.RedisIndex.Clear").
		Info("bumped pod generation", "pod", podEntry.PodIdentifier)
	if r.sweepCh != nil {
		select {
		case r.sweepCh <- struct{}{}:
		default:
		}
	}
	return nil
}

// Sweep walks all request-key hashes in Redis and removes entries whose stamped
// generation is below their pod's current generation. Pages via SCAN to avoid
// blocking the server; HDels stale fields in pipelined batches.
//
// Engine-key entries (prefixed "engine:") are skipped. Empty hashes are deleted.
//
// Returns the count of removed entries.
func (r *RedisIndex) Sweep(ctx context.Context) (int, error) {
	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.Sweep")
	if r.gen.anyClears() == 0 {
		return 0, nil
	}
	gens := r.gen.snapshot()

	const scanBatch int64 = 1024
	removed := 0
	var cursor uint64
	pages := 0
	for {
		keys, next, err := r.RedisClient.Scan(ctx, cursor, "*", scanBatch).Result()
		if err != nil {
			return removed, fmt.Errorf("scan failed: %w", err)
		}
		pages++
		// Filter: skip engine-key entries, keep request-key hashes.
		// Allocate a fresh slice to avoid aliasing the SCAN result buffer.
		filtered := make([]string, 0, len(keys))
		for _, k := range keys {
			if strings.HasPrefix(k, "engine:") {
				continue
			}
			filtered = append(filtered, k)
		}
		if len(filtered) > 0 {
			n, err := r.sweepBatch(ctx, filtered, gens)
			if err != nil {
				return removed, err
			}
			removed += n
		}
		cursor = next
		if cursor == 0 {
			break
		}
	}
	logger.V(1).Info("sweep complete", "pages", pages, "removed", removed)
	if removed > 0 {
		logger.Info("sweep removed stale entries", "removed", removed)
	}
	return removed, nil
}

// sweepBatch processes one SCAN page: HGETALL each, identify stale fields, HDEL
// them in a single pipeline, prune now-empty hashes.
func (r *RedisIndex) sweepBatch(ctx context.Context, keys []string, gens *genCache) (int, error) {
	pipe := r.RedisClient.Pipeline()
	getCmds := make([]*redis.MapStringStringCmd, len(keys))
	for i, k := range keys {
		getCmds[i] = pipe.HGetAll(ctx, k)
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return 0, fmt.Errorf("hgetall pipeline failed: %w", err)
	}

	delPipe := r.RedisClient.Pipeline()
	pruneKeys := make([]string, 0)
	removed := 0
	for i, cmd := range getCmds {
		fields, err := cmd.Result()
		if err != nil || len(fields) == 0 {
			continue
		}
		var stale []string
		for fieldName, valStr := range fields {
			entry, ok := parseRedisPodField(fieldName)
			if !ok {
				continue
			}
			stamped, _ := strconv.ParseUint(valStr, 10, 64)
			if stamped < gens.current(entry.PodIdentifier) {
				stale = append(stale, fieldName)
			}
		}
		if len(stale) == 0 {
			continue
		}
		// HDel takes (key, fields...). Pass as variadic.
		args := make([]any, 0, len(stale))
		for _, s := range stale {
			args = append(args, s)
		}
		delPipe.HDel(ctx, keys[i], stale...)
		removed += len(stale)
		// If we're removing all current fields, mark for prune.
		if len(stale) == len(fields) {
			pruneKeys = append(pruneKeys, keys[i])
		}
	}
	if removed > 0 {
		if _, err := delPipe.Exec(ctx); err != nil {
			return removed, fmt.Errorf("hdel pipeline failed: %w", err)
		}
	}
	// Prune empty hashes.
	for _, k := range pruneKeys {
		if err := pruneRequestKeyScript.Run(ctx, r.RedisClient, []string{k}).Err(); err != nil {
			return removed, fmt.Errorf("prune empty hash: %w", err)
		}
	}
	return removed, nil
}

// StartSweeper runs Sweep on every Clear, debounced by `debounce`. Returns when
// ctx is cancelled.
func (r *RedisIndex) StartSweeper(ctx context.Context, debounce time.Duration) {
	logger := log.FromContext(ctx).WithName("kvblock.RedisIndex.StartSweeper")
	if debounce <= 0 {
		debounce = 100 * time.Millisecond
	}
	if r.sweepCh == nil {
		r.sweepCh = make(chan struct{}, 1)
	}
	for {
		select {
		case <-ctx.Done():
			return
		case <-r.sweepCh:
			timer := time.NewTimer(debounce)
			drained := false
			for !drained {
				select {
				case <-r.sweepCh:
				case <-timer.C:
					drained = true
				case <-ctx.Done():
					timer.Stop()
					return
				}
			}
			if _, err := r.Sweep(ctx); err != nil {
				logger.Error(err, "sweep failed")
			}
		}
	}
}
