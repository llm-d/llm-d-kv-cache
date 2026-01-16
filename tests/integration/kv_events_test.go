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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestPoolWithSubscriberManager demonstrates the integration between
// the Pool and SubscriberManager components.
func TestPoolWithSubscriberManager_Integration(t *testing.T) {
	ctx := context.Background()

	// 1. Create KV index
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	// 2. Create event pool
	poolConfig := kvevents.DefaultConfig()
	poolConfig.Concurrency = 2 // Use 2 workers for testing
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)
	pool.Start(ctx)
	defer pool.Shutdown(ctx)

	// 3. Create subscriber manager
	subscriberManager := kvevents.NewSubscriberManager(pool)
	defer subscriberManager.Shutdown(ctx)

	// 4. Simulate pod reconciliation: add subscribers for "pods"
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

	// 5. Verify subscribers are active
	active, _ := subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 2)
	assert.Contains(t, active, "default/vllm-pod-0")
	assert.Contains(t, active, "default/vllm-pod-1")

	// 6. Simulate pod deletion: remove one subscriber
	subscriberManager.RemoveSubscriber(ctx, "default/vllm-pod-0")
	active, _ = subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 1)
	assert.NotContains(t, active, "default/vllm-pod-0")

	// 7. Simulate pod update with endpoint change
	newEndpoint := "tcp://10.0.0.10:5557"
	err = subscriberManager.EnsureSubscriber(ctx, "default/vllm-pod-1", newEndpoint, "kv@", true)
	require.NoError(t, err)

	// Still one subscriber, but with new endpoint
	active, _ = subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 1)

	// 8. Verify cleanup on shutdown
	subscriberManager.Shutdown(ctx)
	active, _ = subscriberManager.GetActiveSubscribers()
	assert.Len(t, active, 0)
}

// TestSubscriberLifecycle tests the complete lifecycle of a subscriber
// from creation to removal.
func TestSubscriberLifecycle(t *testing.T) {
	ctx := context.Background()

	// Setup
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)
	pool.Start(ctx)
	defer pool.Shutdown(ctx)

	sm := kvevents.NewSubscriberManager(pool)
	defer sm.Shutdown(ctx)

	podID := "default/test-pod"
	endpoint := "tcp://127.0.0.1:5557"

	// Lifecycle: Creation
	t.Run("Creation", func(t *testing.T) {
		err := sm.EnsureSubscriber(ctx, podID, endpoint, "kv@", true)
		assert.NoError(t, err)
		identifiers, _ := sm.GetActiveSubscribers()
		assert.Contains(t, identifiers, podID)
	})

	// Lifecycle: Idempotent creation (same endpoint)
	t.Run("IdempotentCreation", func(t *testing.T) {
		err := sm.EnsureSubscriber(ctx, podID, endpoint, "kv@", true)
		assert.NoError(t, err)
		identifiers, endpoints := sm.GetActiveSubscribers()
		assert.Contains(t, identifiers, podID)
		assert.Equal(t, endpoint, endpoints[0])
	})

	// Lifecycle: Update (different endpoint)
	t.Run("Update", func(t *testing.T) {
		newEndpoint := "tcp://127.0.0.1:5558"
		err := sm.EnsureSubscriber(ctx, podID, newEndpoint, "kv@", true)
		assert.NoError(t, err)
		identifiers, endpoints := sm.GetActiveSubscribers()
		assert.Contains(t, identifiers, podID)
		assert.Equal(t, newEndpoint, endpoints[0])
	})

	// Lifecycle: Removal
	t.Run("Removal", func(t *testing.T) {
		sm.RemoveSubscriber(ctx, podID)
		identifiers, _ := sm.GetActiveSubscribers()
		assert.NotContains(t, identifiers, podID)
	})

	// Lifecycle: Idempotent removal
	t.Run("IdempotentRemoval", func(t *testing.T) {
		sm.RemoveSubscriber(ctx, podID)
		identifiers, _ := sm.GetActiveSubscribers()
		assert.Len(t, identifiers, 0)
	})
}
