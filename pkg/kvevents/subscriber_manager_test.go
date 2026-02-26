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

package kvevents_test

import (
	"context"
	"testing"
	"time"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSubscriberManager_EnsureSubscriber(t *testing.T) {
	ctx := context.Background()

	// Setup pool
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)

	// Create subscriber manager
	sm := kvevents.NewSubscriberManager(pool)

	// Test adding a subscriber
	podID := "default/test-pod-0"
	endpoint := "tcp://127.0.0.1:5557"
	topicFilter := "kv@"

	err = sm.EnsureSubscriber(ctx, podID, endpoint, topicFilter, true)
	assert.NoError(t, err)

	// Verify subscriber was added
	identifiers, endpoints := sm.GetActiveSubscribers()
	assert.Contains(t, identifiers, podID)
	assert.Len(t, identifiers, 1)
	assert.Contains(t, endpoints, endpoint)

	// Ensure with same endpoint should be no-op
	err = sm.EnsureSubscriber(ctx, podID, endpoint, topicFilter, true)
	assert.NoError(t, err)
	identifiers, _ = sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 1)

	// Shutdown
	sm.Shutdown(ctx)
	identifiers, _ = sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 0)
}

func TestSubscriberManager_RemoveSubscriber(t *testing.T) {
	ctx := context.Background()

	// Setup pool
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)

	// Create subscriber manager
	sm := kvevents.NewSubscriberManager(pool)

	// Add subscriber
	podID := "default/test-pod-0"
	endpoint := "tcp://127.0.0.1:5557"
	topicFilter := "kv@"

	err = sm.EnsureSubscriber(ctx, podID, endpoint, topicFilter, true)
	require.NoError(t, err)

	// Remove subscriber
	sm.RemoveSubscriber(ctx, podID)
	identifiers, _ := sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 0)

	// Remove again should be no-op
	sm.RemoveSubscriber(ctx, podID)
	identifiers, _ = sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 0)
}

func TestSubscriberManager_MultipleSubscribers(t *testing.T) {
	ctx := context.Background()

	// Setup pool
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)

	// Create subscriber manager
	sm := kvevents.NewSubscriberManager(pool)

	// Add multiple subscribers
	pods := []struct {
		id       string
		endpoint string
	}{
		{"default/pod-0", "tcp://10.0.0.1:5557"},
		{"default/pod-1", "tcp://10.0.0.2:5557"},
		{"default/pod-2", "tcp://10.0.0.3:5557"},
	}

	for _, pod := range pods {
		err := sm.EnsureSubscriber(ctx, pod.id, pod.endpoint, "kv@", true)
		require.NoError(t, err)
	}

	// Verify all subscribers were added
	identifiers, endpoints := sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 3)
	for _, pod := range pods {
		assert.Contains(t, identifiers, pod.id)
	}
	assert.Len(t, endpoints, 3)
	for _, pod := range pods {
		assert.Contains(t, endpoints, pod.endpoint)
	}

	// Remove one subscriber
	sm.RemoveSubscriber(ctx, "default/pod-1")
	identifiers, _ = sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 2)
	assert.NotContains(t, identifiers, "default/pod-1")

	// Shutdown remaining subscribers
	sm.Shutdown(ctx)
	identifiers, _ = sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 0)
}

func TestSubscriberManager_EndpointChange(t *testing.T) {
	ctx := context.Background()

	// Setup pool
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)

	// Create subscriber manager
	sm := kvevents.NewSubscriberManager(pool)

	podID := "default/test-pod-0"
	endpoint1 := "tcp://10.0.0.1:5557"
	endpoint2 := "tcp://10.0.0.2:5557"

	// Add subscriber with first endpoint
	err = sm.EnsureSubscriber(ctx, podID, endpoint1, "kv@", true)
	require.NoError(t, err)
	identifiers, _ := sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 1)

	// Change endpoint
	err = sm.EnsureSubscriber(ctx, podID, endpoint2, "kv@", true)
	require.NoError(t, err)

	// Should still have one subscriber (old was removed, new was added)
	identifiers, endpoints := sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 1)
	assert.Contains(t, identifiers, podID)
	assert.Len(t, endpoints, 1)
	assert.Contains(t, endpoints, endpoint2)

	// Shutdown
	sm.Shutdown(ctx)
	identifiers, _ = sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 0)
}

func TestSubscriberManager_ConcurrentOperations(t *testing.T) {
	ctx := context.Background()

	// Setup pool
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	require.NoError(t, err)

	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)

	// Create subscriber manager
	sm := kvevents.NewSubscriberManager(pool)

	// Concurrently add subscribers
	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func(id int) {
			defer func() { done <- true }()
			podID := "default/pod-" + string(rune('0'+id))
			endpoint := "tcp://10.0.0." + string(rune('0'+id)) + ":5557"
			if err := sm.EnsureSubscriber(ctx, podID, endpoint, "kv@", true); err != nil {
				t.Errorf("failed to add subscriber %s: %v", podID, err)
			}
		}(i)
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// Should have 10 subscribers
	time.Sleep(100 * time.Millisecond) // Give time for subscribers to start
	identifiers, _ := sm.GetActiveSubscribers()
	assert.Len(t, identifiers, 10)

	sm.Shutdown(ctx)
}
