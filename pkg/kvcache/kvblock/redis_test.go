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

package kvblock_test

import (
	"encoding/json"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	. "github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// createRedisIndexForTesting creates a new RedisIndex with a mock Redis server for testing.
func createRedisIndexForTesting(t *testing.T) Index {
	t.Helper()
	return createRedisIndexForInspection(t)
}

func createRedisIndexForInspection(t *testing.T) *RedisIndex {
	t.Helper()
	server, err := miniredis.Run()
	require.NoError(t, err)

	// Store server reference for cleanup
	t.Cleanup(func() {
		server.Close()
	})

	redisConfig := &RedisIndexConfig{
		Address: server.Addr(),
	}
	index, err := NewRedisIndex(redisConfig)
	require.NoError(t, err)
	redisIndex, ok := index.(*RedisIndex)
	require.True(t, ok)
	return redisIndex
}

// TestRedisIndexBehavior tests the Redis index implementation using common test behaviors.
func TestRedisIndexBehavior(t *testing.T) {
	testCommonIndexBehavior(t, createRedisIndexForTesting)
}

func TestRedisClearPreservesUnrelatedKeys(t *testing.T) {
	index := createRedisIndexForInspection(t)
	ctx := t.Context()
	pod := PodEntry{PodIdentifier: "pod-clear", DeviceTier: "gpu"}
	requestKey := BlockHash(0xC1EA1001)
	unrelatedField := redisTestPodField(t, pod)

	require.NoError(t, index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{pod}))
	require.NoError(t, index.RedisClient.HSet(ctx, "unrelated-hash", unrelatedField, "keep").Err())
	require.NoError(t, index.RedisClient.Set(ctx, "unrelated-string", "keep", 0).Err())

	require.NoError(t, index.Clear(ctx, pod.PodIdentifier))

	gotHash, err := index.RedisClient.HGet(ctx, "unrelated-hash", unrelatedField).Result()
	require.NoError(t, err)
	require.Equal(t, "keep", gotHash)

	gotString, err := index.RedisClient.Get(ctx, "unrelated-string").Result()
	require.NoError(t, err)
	require.Equal(t, "keep", gotString)

	hits, err := index.Lookup(ctx, []BlockHash{requestKey}, nil)
	require.NoError(t, err)
	require.Empty(t, hits[requestKey])

	exists, err := index.RedisClient.Exists(ctx, requestKey.String()).Result()
	require.NoError(t, err)
	require.Zero(t, exists)
}

func TestRedisEvictPrunesReverseIndex(t *testing.T) {
	index := createRedisIndexForInspection(t)
	ctx := t.Context()
	pod := PodEntry{PodIdentifier: "pod-evict", DeviceTier: "gpu"}
	requestKey := BlockHash(0xC1EA1002)
	podEntriesKey := "kvblock:pod:" + pod.PodIdentifier + ":entries"

	require.NoError(t, index.Add(ctx, nil, []BlockHash{requestKey}, []PodEntry{pod}))
	count, err := index.RedisClient.SCard(ctx, podEntriesKey).Result()
	require.NoError(t, err)
	require.EqualValues(t, 1, count)

	require.NoError(t, index.Evict(ctx, requestKey, RequestKey, []PodEntry{pod}))

	count, err = index.RedisClient.SCard(ctx, podEntriesKey).Result()
	require.NoError(t, err)
	require.Zero(t, count)
}

func TestRedisClearDropsMalformedReverseEntry(t *testing.T) {
	index := createRedisIndexForInspection(t)
	ctx := t.Context()
	podEntriesKey := "kvblock:pod:pod-malformed:entries"

	require.NoError(t, index.RedisClient.SAdd(ctx, podEntriesKey, "malformed").Err())
	require.NoError(t, index.Clear(ctx, "pod-malformed"))

	count, err := index.RedisClient.SCard(ctx, podEntriesKey).Result()
	require.NoError(t, err)
	require.Zero(t, count)
}

func redisTestPodField(t *testing.T, entry PodEntry) string {
	t.Helper()
	value, err := json.Marshal(entry)
	require.NoError(t, err)
	return string(value)
}
