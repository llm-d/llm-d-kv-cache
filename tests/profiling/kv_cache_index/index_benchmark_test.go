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

package main

import (
	"context"
	"math/rand/v2"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	modelName = "bert-base-uncased"
	// Default number of keys for benchmarking.
	benchNumKeys = 10000
)

// generateWorkloadKeys creates a slice of keys with random chunk hashes.
func generateWorkloadKeys(numKeys int) []kvblock.Key {
	// Use a fixed seed to ensure the exact same keys are generated for all profiling sessions.
	// This ensures we are comparing index implementations on identical data.
	//nolint:gosec // Weak RNG is acceptable for benchmarking.
	randGen := rand.New(rand.NewPCG(42, 1024))

	keys := make([]kvblock.Key, numKeys)
	for i := range numKeys {
		keys[i] = kvblock.Key{
			ModelName: modelName,
			ChunkHash: randGen.Uint64(),
		}
	}
	return keys
}

// helper to initialize specific index types.
// redisAddr is optional; only used if indexType is "redis".
func getIndexConfig(indexType, redisAddr string) *kvblock.IndexConfig {
	switch indexType {
	case "redis":
		cfg := kvblock.DefaultRedisIndexConfig()
		cfg.Address = redisAddr
		return &kvblock.IndexConfig{
			RedisConfig:   cfg,
			EnableMetrics: false,
		}
	case "cost":
		return &kvblock.IndexConfig{
			CostAwareMemoryConfig: kvblock.DefaultCostAwareMemoryIndexConfig(),
			EnableMetrics:         false,
		}
	case "memory":
		return kvblock.DefaultIndexConfig()
	default:
		return kvblock.DefaultIndexConfig()
	}
}

// setupMiniredis starts a purely in-memory redis instance.
// Returns the instance and a cleanup function.
//
//nolint:gocritic // Unnamed results are preferred by the linter configuration.
func setupMiniredis(b *testing.B) (*miniredis.Miniredis, func()) {
	b.Helper()
	s, err := miniredis.Run()
	if err != nil {
		b.Fatalf("failed to start miniredis: %v", err)
	}
	return s, func() { s.Close() }
}

// benchmarkAdd measures the performance of Adding keys to the index.
func benchmarkAdd(b *testing.B, indexType string) {
	b.Helper()
	ctx := context.Background()
	podEntries := []kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}
	keys := generateWorkloadKeys(benchNumKeys)

	var redisAddr string

	// Clean setup for Miniredis specifically
	if indexType == "redis" {
		mr, cleanup := setupMiniredis(b)
		defer cleanup()
		redisAddr = mr.Addr()
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		// Create a fresh index client, but connect to the SAME background redis server
		cfg := getIndexConfig(indexType, redisAddr)
		index, err := kvblock.NewIndex(ctx, cfg)
		if err != nil {
			b.Fatalf("failed to create index: %v", err)
		}

		b.StartTimer()

		// Pass 'keys' for both engineKeys and requestKeys
		err = index.Add(ctx, keys, keys, podEntries)
		if err != nil {
			b.Fatalf("failed to add entries: %v", err)
		}
	}
}

// benchmarkLookup measures the performance of Looking up keys.
func benchmarkLookup(b *testing.B, indexType string) {
	b.Helper()
	ctx := context.Background()
	podEntries := []kvblock.PodEntry{{PodIdentifier: "pod1", DeviceTier: "gpu"}}

	// Intentionally use an empty podIdentifierSet to return all pods during lookup,
	// as documented in the Index interface.
	podIdentifierSet := sets.Set[string]{}

	keys := generateWorkloadKeys(benchNumKeys)

	var redisAddr string
	if indexType == "redis" {
		mr, cleanup := setupMiniredis(b)
		defer cleanup()
		redisAddr = mr.Addr()
	}

	// Setup: Create index and populate it
	cfg := getIndexConfig(indexType, redisAddr)
	index, err := kvblock.NewIndex(ctx, cfg)
	if err != nil {
		b.Fatalf("failed to create index: %v", err)
	}

	if err := index.Add(ctx, keys, keys, podEntries); err != nil {
		b.Fatalf("failed to populate index: %v", err)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, err = index.Lookup(ctx, keys, podIdentifierSet)
		if err != nil {
			b.Fatalf("failed to lookup entries: %v", err)
		}
	}
}

// --- Benchmark Entry Points ---

func BenchmarkInMemory_Add(b *testing.B) {
	benchmarkAdd(b, "memory")
}

func BenchmarkInMemory_Lookup(b *testing.B) {
	benchmarkLookup(b, "memory")
}

func BenchmarkRedis_Add(b *testing.B) {
	benchmarkAdd(b, "redis")
}

func BenchmarkRedis_Lookup(b *testing.B) {
	benchmarkLookup(b, "redis")
}

func BenchmarkCostAware_Add(b *testing.B) {
	benchmarkAdd(b, "cost")
}

func BenchmarkCostAware_Lookup(b *testing.B) {
	benchmarkLookup(b, "cost")
}
