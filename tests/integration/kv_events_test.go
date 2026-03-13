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

package integration_test

import (
	"context"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPoolWithSubscriberManager demonstrates the integration between
// the Pool and SubscriberManager components.
func TestPoolWithSubscriberManager_Integration(t *testing.T) {
	ctx := context.Background()

	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	poolConfig.Concurrency = 2
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)
	defer pool.Shutdown(ctx)

	subscriberManager := kvevents.NewSubscriberManager(pool)
	defer subscriberManager.Shutdown(ctx)

	pods := []struct {
		id       string
		endpoint string
	}{
		{"default/vllm-pod-0", "tcp://10.0.0.1:5557"},
		{"default/vllm-pod-1", "tcp://10.0.0.2:5557"},
	}

	for _, pod := range pods {
		err := subscriberManager.EnsureSubscriber(ctx, pod.id, pod.endpoint, "kv@", true)
		require.NoError(t, err)
	}

	active, _ := subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 2)
	assert.Contains(t, active, "default/vllm-pod-0")
	assert.Contains(t, active, "default/vllm-pod-1")

	subscriberManager.RemoveSubscriber(ctx, "default/vllm-pod-0")
	active, _ = subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 1)
	assert.NotContains(t, active, "default/vllm-pod-0")

	newEndpoint := "tcp://10.0.0.10:5557"
	err = subscriberManager.EnsureSubscriber(ctx, "default/vllm-pod-1", newEndpoint, "kv@", true)
	require.NoError(t, err)

	active, _ = subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 1)

	subscriberManager.Shutdown(ctx)
	active, _ = subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 0)
}

// TestSubscriberLifecycle tests the complete lifecycle of a subscriber
// from creation to removal.
func TestSubscriberLifecycle(t *testing.T) {
	ctx := context.Background()

	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	require.NoError(t, err)
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)
	defer pool.Shutdown(ctx)

	sm := kvevents.NewSubscriberManager(pool)
	defer sm.Shutdown(ctx)

	podID := "default/test-pod"
	endpoint := "tcp://127.0.0.1:5557"

	t.Run("Creation", func(t *testing.T) {
		err := sm.EnsureSubscriber(ctx, podID, endpoint, "kv@", true)
		assert.NoError(t, err)
		identifiers, _ := sm.GetActiveSubscribers()
		assert.Contains(t, identifiers, podID)
	})

	t.Run("IdempotentCreation", func(t *testing.T) {
		err := sm.EnsureSubscriber(ctx, podID, endpoint, "kv@", true)
		assert.NoError(t, err)
		identifiers, endpoints := sm.GetActiveSubscribers()
		assert.Contains(t, identifiers, podID)
		assert.Equal(t, endpoint, endpoints[0])
	})

	t.Run("Update", func(t *testing.T) {
		newEndpoint := "tcp://127.0.0.1:5558"
		err := sm.EnsureSubscriber(ctx, podID, newEndpoint, "kv@", true)
		assert.NoError(t, err)
		identifiers, endpoints := sm.GetActiveSubscribers()
		assert.Contains(t, identifiers, podID)
		assert.Equal(t, newEndpoint, endpoints[0])
	})

	t.Run("Removal", func(t *testing.T) {
		sm.RemoveSubscriber(ctx, podID)
		identifiers, _ := sm.GetActiveSubscribers()
		assert.NotContains(t, identifiers, podID)
	})

	t.Run("IdempotentRemoval", func(t *testing.T) {
		sm.RemoveSubscriber(ctx, podID)
		identifiers, _ := sm.GetActiveSubscribers()
		assert.Len(t, identifiers, 0)
	})
}
