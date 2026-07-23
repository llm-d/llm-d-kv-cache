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
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/redis/go-redis/v9"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	redisClearTargetPod     = "pod-0"
	redisClearSeedBatchSize = 1024
)

type redisClearBenchmarkCase struct {
	name          string
	requestKeys   int
	podCount      int
	unrelatedKeys int
}

var (
	redisClearScaleCases = []redisClearBenchmarkCase{
		{name: "targetKeys=512/unrelatedKeys=512/pods=8", requestKeys: 512, podCount: 8, unrelatedKeys: 512},
		{name: "targetKeys=2048/unrelatedKeys=2048/pods=8", requestKeys: 2048, podCount: 8, unrelatedKeys: 2048},
		{name: "targetKeys=8192/unrelatedKeys=8192/pods=8", requestKeys: 8192, podCount: 8, unrelatedKeys: 8192},
		{name: "targetKeys=32768/unrelatedKeys=32768/pods=8", requestKeys: 32768, podCount: 8, unrelatedKeys: 32768},
	}
	redisClearSharedKeyspaceCases = []redisClearBenchmarkCase{
		{name: "targetKeys=2048/unrelatedKeys=0/pods=8", requestKeys: 2048, podCount: 8},
		{name: "targetKeys=2048/unrelatedKeys=8192/pods=8", requestKeys: 2048, podCount: 8, unrelatedKeys: 8192},
		{name: "targetKeys=2048/unrelatedKeys=32768/pods=8", requestKeys: 2048, podCount: 8, unrelatedKeys: 32768},
	}
	redisClearPodFanoutCases = []redisClearBenchmarkCase{
		{name: "targetKeys=2048/unrelatedKeys=8192/pods=1", requestKeys: 2048, podCount: 1, unrelatedKeys: 8192},
		{name: "targetKeys=2048/unrelatedKeys=8192/pods=8", requestKeys: 2048, podCount: 8, unrelatedKeys: 8192},
		{name: "targetKeys=2048/unrelatedKeys=8192/pods=32", requestKeys: 2048, podCount: 32, unrelatedKeys: 8192},
	}
)

func BenchmarkRedisClearScaleLegacyFullScan(b *testing.B) {
	benchmarkRedisClearCases(b, false, redisClearScaleCases)
}

func BenchmarkRedisClearScaleReverseIndex(b *testing.B) {
	benchmarkRedisClearCases(b, true, redisClearScaleCases)
}

func BenchmarkRedisClearSharedKeyspaceLegacyFullScan(b *testing.B) {
	benchmarkRedisClearCases(b, false, redisClearSharedKeyspaceCases)
}

func BenchmarkRedisClearSharedKeyspaceReverseIndex(b *testing.B) {
	benchmarkRedisClearCases(b, true, redisClearSharedKeyspaceCases)
}

func BenchmarkRedisClearPodFanoutLegacyFullScan(b *testing.B) {
	benchmarkRedisClearCases(b, false, redisClearPodFanoutCases)
}

func BenchmarkRedisClearPodFanoutReverseIndex(b *testing.B) {
	benchmarkRedisClearCases(b, true, redisClearPodFanoutCases)
}

func benchmarkRedisClearCases(b *testing.B, reverseIndex bool, cases []redisClearBenchmarkCase) {
	b.Helper()
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			benchmarkRedisClear(b, tc, reverseIndex)
		})
	}
}

func benchmarkRedisClear(b *testing.B, tc redisClearBenchmarkCase, reverseIndex bool) {
	b.Helper()
	server, cleanup := setupMiniredis(b)
	b.Cleanup(cleanup)
	benchmarkRedisClearWithAddress(b, server.Addr(), tc, reverseIndex)
}

func benchmarkRedisClearWithAddress(b *testing.B, address string, tc redisClearBenchmarkCase, reverseIndex bool) {
	b.Helper()
	index, err := kvblock.NewRedisIndex(&kvblock.RedisIndexConfig{Address: address})
	if err != nil {
		b.Fatalf("failed to create redis index: %v", err)
	}
	redisIndex, ok := index.(*kvblock.RedisIndex)
	if !ok {
		b.Fatalf("unexpected index type %T", index)
	}
	ctx := log.IntoContext(context.Background(), logr.Discard())
	b.Cleanup(func() {
		if err := redisIndex.RedisClient.FlushDB(ctx).Err(); err != nil {
			b.Fatalf("failed to flush redis during cleanup: %v", err)
		}
		if err := redisIndex.RedisClient.Close(); err != nil {
			b.Fatalf("failed to close redis client: %v", err)
		}
	})

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		if err := redisIndex.RedisClient.FlushDB(ctx).Err(); err != nil {
			b.Fatalf("failed to flush redis: %v", err)
		}
		seedRedisClearData(b, ctx, redisIndex, tc, reverseIndex)
		b.StartTimer()

		if reverseIndex {
			err = redisIndex.Clear(ctx, redisClearTargetPod)
		} else {
			err = legacyRedisClear(ctx, redisIndex.RedisClient, redisClearTargetPod)
		}
		if err != nil {
			b.Fatalf("failed to clear redis data: %v", err)
		}
	}
}

func seedRedisClearData(
	b *testing.B,
	ctx context.Context,
	index *kvblock.RedisIndex,
	tc redisClearBenchmarkCase,
	reverseIndex bool,
) {
	b.Helper()
	if reverseIndex {
		seedRedisClearReverseIndexData(b, ctx, index, tc)
	} else {
		seedRedisClearLegacyData(b, ctx, index, tc)
	}
	seedUnrelatedRedisData(b, ctx, index, tc.unrelatedKeys)
}

func seedRedisClearReverseIndexData(
	b *testing.B,
	ctx context.Context,
	index *kvblock.RedisIndex,
	tc redisClearBenchmarkCase,
) {
	b.Helper()
	requestKeys := make([]kvblock.BlockHash, tc.requestKeys)
	for key := range requestKeys {
		requestKeys[key] = redisClearBlockHash(b, key+1)
	}
	for pod := 0; pod < tc.podCount; pod++ {
		entry := kvblock.PodEntry{PodIdentifier: fmt.Sprintf("pod-%d", pod), DeviceTier: "gpu"}
		for start := 0; start < len(requestKeys); start += redisClearSeedBatchSize {
			end := min(start+redisClearSeedBatchSize, len(requestKeys))
			if err := index.Add(ctx, nil, requestKeys[start:end], []kvblock.PodEntry{entry}); err != nil {
				b.Fatalf("failed to seed redis data through Add: %v", err)
			}
		}
	}
}

func seedRedisClearLegacyData(
	b *testing.B,
	ctx context.Context,
	index *kvblock.RedisIndex,
	tc redisClearBenchmarkCase,
) {
	b.Helper()
	pipe := index.RedisClient.Pipeline()
	pending := 0
	for key := 0; key < tc.requestKeys; key++ {
		redisKey := redisClearBlockHash(b, key+1).String()
		for pod := 0; pod < tc.podCount; pod++ {
			entry := kvblock.PodEntry{PodIdentifier: fmt.Sprintf("pod-%d", pod), DeviceTier: "gpu"}
			field := redisClearPodField(b, entry)
			pipe.HSet(ctx, redisKey, field, "")
			pending++
			if pending == redisClearSeedBatchSize {
				execRedisClearSeedPipeline(b, ctx, pipe)
				pipe = index.RedisClient.Pipeline()
				pending = 0
			}
		}
	}
	if pending > 0 {
		execRedisClearSeedPipeline(b, ctx, pipe)
	}
}

func seedUnrelatedRedisData(b *testing.B, ctx context.Context, index *kvblock.RedisIndex, count int) {
	b.Helper()
	if count == 0 {
		return
	}
	pipe := index.RedisClient.Pipeline()
	pending := 0
	for key := 0; key < count; key++ {
		field := redisClearPodField(b, kvblock.PodEntry{PodIdentifier: "other", DeviceTier: "gpu"})
		pipe.HSet(ctx, fmt.Sprintf("unrelated:%d", key), field, "")
		pending++
		if pending == redisClearSeedBatchSize {
			execRedisClearSeedPipeline(b, ctx, pipe)
			pipe = index.RedisClient.Pipeline()
			pending = 0
		}
	}
	if pending > 0 {
		execRedisClearSeedPipeline(b, ctx, pipe)
	}
}

func execRedisClearSeedPipeline(b *testing.B, ctx context.Context, pipe redis.Pipeliner) {
	b.Helper()
	if _, err := pipe.Exec(ctx); err != nil {
		b.Fatalf("failed to seed redis data: %v", err)
	}
}

func legacyRedisClear(ctx context.Context, client *redis.Client, podIdentifier string) error {
	const scanBatch int64 = 1024
	var cursor uint64
	for {
		keys, next, err := client.Scan(ctx, cursor, "*", scanBatch).Result()
		if err != nil {
			return err
		}
		for _, key := range keys {
			if strings.HasPrefix(key, "engine:") {
				continue
			}
			fields, err := client.HKeys(ctx, key).Result()
			if err != nil {
				return err
			}
			var stale []string
			for _, field := range fields {
				var entry kvblock.PodEntry
				if json.Unmarshal([]byte(field), &entry) == nil && entry.PodIdentifier == podIdentifier {
					stale = append(stale, field)
				}
			}
			if len(stale) > 0 {
				if err := client.HDel(ctx, key, stale...).Err(); err != nil {
					return err
				}
			}
		}
		if cursor = next; cursor == 0 {
			return nil
		}
	}
}

func redisClearBlockHash(b *testing.B, key int) kvblock.BlockHash {
	b.Helper()
	value, err := strconv.ParseUint(strconv.FormatInt(int64(key), 10), 10, 64)
	if err != nil {
		b.Fatalf("failed to convert benchmark key: %v", err)
	}
	return kvblock.BlockHash(value)
}

func redisClearPodField(b *testing.B, entry kvblock.PodEntry) string {
	b.Helper()
	value, err := json.Marshal(entry)
	if err != nil {
		b.Fatalf("failed to encode pod entry: %v", err)
	}
	return string(value)
}
