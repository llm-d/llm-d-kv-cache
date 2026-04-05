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

//nolint:testpackage // need to test unexported processEventBatch and internal fields
package kvevents

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// --- Mock Implementations ---

type mockIndex struct {
	mu          sync.Mutex
	addCalls    []addCall
	evictCalls  []evictCall
	requestKeys map[kvblock.BlockHash]kvblock.BlockHash // engineKey → requestKey
	addErr      error
	evictErr    error
}

type addCall struct {
	engineKeys  []kvblock.BlockHash
	requestKeys []kvblock.BlockHash
	entries     []kvblock.PodEntry
}

type evictCall struct {
	key     kvblock.BlockHash
	keyType kvblock.KeyType
	entries []kvblock.PodEntry
}

func newMockIndex() *mockIndex {
	return &mockIndex{
		requestKeys: make(map[kvblock.BlockHash]kvblock.BlockHash),
	}
}

func (m *mockIndex) Add(_ context.Context, engineKeys, requestKeys []kvblock.BlockHash, entries []kvblock.PodEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.addCalls = append(m.addCalls, addCall{
		engineKeys:  engineKeys,
		requestKeys: requestKeys,
		entries:     entries,
	})
	return m.addErr
}

func (m *mockIndex) Evict(_ context.Context, key kvblock.BlockHash, keyType kvblock.KeyType, entries []kvblock.PodEntry) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.evictCalls = append(m.evictCalls, evictCall{
		key:     key,
		keyType: keyType,
		entries: entries,
	})
	return m.evictErr
}

func (m *mockIndex) Lookup(
	_ context.Context, _ []kvblock.BlockHash, _ sets.Set[string],
) (map[kvblock.BlockHash][]kvblock.PodEntry, error) {
	return nil, nil //nolint:nilnil // mock stub, no real lookup needed
}

func (m *mockIndex) GetRequestKey(_ context.Context, engineKey kvblock.BlockHash) (kvblock.BlockHash, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if rk, ok := m.requestKeys[engineKey]; ok {
		return rk, nil
	}
	return kvblock.EmptyBlockHash, fmt.Errorf("engine key %v not found", engineKey)
}

func (m *mockIndex) getAddCalls() []addCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]addCall, len(m.addCalls))
	copy(out, m.addCalls)
	return out
}

func (m *mockIndex) getEvictCalls() []evictCall {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]evictCall, len(m.evictCalls))
	copy(out, m.evictCalls)
	return out
}

type mockTokenProcessor struct {
	returnKeys []kvblock.BlockHash
}

func (m *mockTokenProcessor) TokensToKVBlockKeys(_ kvblock.BlockHash, _ []uint32, _ string) []kvblock.BlockHash {
	return m.returnKeys
}

type mockAdapter struct {
	parseFunc func(msg *RawMessage) (string, string, EventBatch, error)
}

//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (m *mockAdapter) ParseMessage(msg *RawMessage) (string, string, EventBatch, error) {
	return m.parseFunc(msg)
}

func (m *mockAdapter) ShardingKey(msg *RawMessage) string {
	return msg.Topic
}

// --- Tests ---

func TestNewPool(t *testing.T) {
	t.Run("nil config uses defaults", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{}
		adapter := &mockAdapter{}

		p := NewPool(nil, idx, tp, adapter)

		assert.Equal(t, 4, p.concurrency)
		assert.Len(t, p.queues, 4)
	})

	t.Run("custom concurrency", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{}
		adapter := &mockAdapter{}

		cfg := &Config{Concurrency: 8}
		p := NewPool(cfg, idx, tp, adapter)

		assert.Equal(t, 8, p.concurrency)
		assert.Len(t, p.queues, 8)
	})
}

func TestPool_ProcessEventBatch_BlockStored(t *testing.T) {
	t.Run("basic block stored with default GPU tier", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{
			returnKeys: []kvblock.BlockHash{100, 200, 300},
		}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{9001, 9002, 9003},
					Tokens:      []uint32{128000, 2675, 527, 264},
					ParentHash:  0,
					DeviceTier:  "",
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getAddCalls()
		require.Len(t, calls, 1)
		assert.Equal(t, []kvblock.BlockHash{9001, 9002, 9003}, calls[0].engineKeys)
		assert.Equal(t, []kvblock.BlockHash{100, 200, 300}, calls[0].requestKeys)
		assert.Equal(t, []kvblock.PodEntry{{PodIdentifier: "default/pod-0", DeviceTier: "GPU"}}, calls[0].entries)
	})

	t.Run("custom device tier lowercased", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{9001},
					Tokens:      []uint32{1, 2},
					DeviceTier:  "CPU",
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getAddCalls()
		require.Len(t, calls, 1)
		assert.Equal(t, "cpu", calls[0].entries[0].DeviceTier)
	})

	t.Run("with LoRA name", func(t *testing.T) {
		idx := newMockIndex()
		var capturedModel string
		tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}
		// Override to capture the model name
		customTP := &captureModelTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

		p := NewPool(&Config{Concurrency: 1}, idx, customTP, &mockAdapter{})

		loraName := "my-lora-adapter"
		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{9001},
					Tokens:      []uint32{1, 2},
					LoraName:    &loraName,
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")
		_ = capturedModel
		_ = tp

		assert.Equal(t, "my-lora-adapter", customTP.lastModelName)
	})

	t.Run("with parent hash", func(t *testing.T) {
		idx := newMockIndex()
		idx.requestKeys[kvblock.BlockHash(5555)] = kvblock.BlockHash(42)

		customTP := &captureModelTokenProcessor{returnKeys: []kvblock.BlockHash{100}}
		p := NewPool(&Config{Concurrency: 1}, idx, customTP, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{9001},
					Tokens:      []uint32{1, 2},
					ParentHash:  5555,
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		assert.Equal(t, kvblock.BlockHash(42), customTP.lastParentKey)
	})

	t.Run("parent hash not found skips event", func(t *testing.T) {
		idx := newMockIndex() // no requestKeys registered
		tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{9001},
					Tokens:      []uint32{1, 2},
					ParentHash:  9999, // not in index
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getAddCalls()
		assert.Empty(t, calls)
	})

	t.Run("empty block hashes skips Add", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{},
					Tokens:      []uint32{1, 2},
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getAddCalls()
		assert.Empty(t, calls)
	})

	t.Run("Add error does not stop batch processing", func(t *testing.T) {
		idx := newMockIndex()
		idx.addErr = fmt.Errorf("add failed")
		tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockStoredEvent{
					BlockHashes: []uint64{9001},
					Tokens:      []uint32{1, 2},
				},
				&BlockStoredEvent{
					BlockHashes: []uint64{9002},
					Tokens:      []uint32{3, 4},
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		// Both events attempted despite first failing
		calls := idx.getAddCalls()
		assert.Len(t, calls, 2)
	})
}

func TestPool_ProcessEventBatch_BlockRemoved(t *testing.T) {
	t.Run("basic block removed", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{9001, 9002},
					DeviceTier:  "GPU",
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getEvictCalls()
		require.Len(t, calls, 2)
		assert.Equal(t, kvblock.BlockHash(9001), calls[0].key)
		assert.Equal(t, kvblock.EngineKey, calls[0].keyType)
		assert.Equal(t, "gpu", calls[0].entries[0].DeviceTier)
		assert.Equal(t, kvblock.BlockHash(9002), calls[1].key)
	})

	t.Run("default device tier when empty", func(t *testing.T) {
		idx := newMockIndex()
		tp := &mockTokenProcessor{}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{9001},
					DeviceTier:  "",
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getEvictCalls()
		require.Len(t, calls, 1)
		assert.Equal(t, "GPU", calls[0].entries[0].DeviceTier)
	})

	t.Run("Evict error does not stop processing", func(t *testing.T) {
		idx := newMockIndex()
		idx.evictErr = fmt.Errorf("evict failed")
		tp := &mockTokenProcessor{}

		p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

		batch := &EventBatch{
			Events: []GenericEvent{
				&BlockRemovedEvent{
					BlockHashes: []uint64{9001, 9002, 9003},
				},
			},
		}

		p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

		calls := idx.getEvictCalls()
		assert.Len(t, calls, 3)
	})
}

func TestPool_ProcessEventBatch_AllBlocksCleared(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{}

	p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

	batch := &EventBatch{
		Events: []GenericEvent{
			&AllBlocksClearedEvent{DeviceTier: "GPU"},
		},
	}

	// Should not panic, and no Add/Evict calls
	p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

	assert.Empty(t, idx.getAddCalls())
	assert.Empty(t, idx.getEvictCalls())
}

func TestPool_ProcessEventBatch_MixedEvents(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

	p := NewPool(&Config{Concurrency: 1}, idx, tp, &mockAdapter{})

	batch := &EventBatch{
		Events: []GenericEvent{
			&BlockStoredEvent{
				BlockHashes: []uint64{9001},
				Tokens:      []uint32{1, 2},
			},
			&BlockRemovedEvent{
				BlockHashes: []uint64{8001},
			},
			&AllBlocksClearedEvent{DeviceTier: "GPU"},
			&BlockStoredEvent{
				BlockHashes: []uint64{9002},
				Tokens:      []uint32{3, 4},
			},
		},
	}

	p.processEventBatch(context.Background(), batch, "default/pod-0", "llama-3")

	assert.Len(t, idx.getAddCalls(), 2)
	assert.Len(t, idx.getEvictCalls(), 1)
}

func TestPool_AddTask_Sharding(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{}
	adapter := &mockAdapter{}

	p := NewPool(&Config{Concurrency: 4}, idx, tp, adapter)

	// Messages with the same topic should always go to the same queue.
	for i := 0; i < 100; i++ {
		p.AddTask(&RawMessage{Topic: "pod-a", Payload: []byte{byte(i)}})
	}

	// Count how many queues have items
	nonEmpty := 0
	for _, q := range p.queues {
		if q.Len() > 0 {
			nonEmpty++
		}
	}

	// Same sharding key → all 100 messages go to exactly one queue
	assert.Equal(t, 1, nonEmpty)

	// Different keys may spread across queues
	for i := 0; i < 100; i++ {
		p.AddTask(&RawMessage{Topic: fmt.Sprintf("pod-%d", i)})
	}
}

func TestPool_StartAndShutdown(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{}
	adapter := &mockAdapter{}

	p := NewPool(&Config{Concurrency: 2}, idx, tp, adapter)

	ctx, cancel := context.WithCancel(context.Background())
	p.Start(ctx)

	// Allow workers to start
	time.Sleep(50 * time.Millisecond)

	cancel()
	p.Shutdown(ctx)
}

func TestPool_EndToEnd(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100, 200}}

	adapter := &mockAdapter{
		parseFunc: func(msg *RawMessage) (string, string, EventBatch, error) {
			return "default/pod-0", "llama-3", EventBatch{
				Events: []GenericEvent{
					&BlockStoredEvent{
						BlockHashes: []uint64{9001, 9002},
						Tokens:      []uint32{128000, 2675, 527, 264},
						DeviceTier:  "GPU",
					},
				},
			}, nil
		},
	}

	cfg := &Config{Concurrency: 2}
	pool := NewPool(cfg, idx, tp, adapter)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool.Start(ctx)

	pool.AddTask(&RawMessage{
		Topic:   "kv@default/pod-0@llama-3",
		Payload: []byte("test"),
	})

	// Wait for async processing
	require.Eventually(t, func() bool {
		return len(idx.getAddCalls()) == 1
	}, 2*time.Second, 10*time.Millisecond)

	calls := idx.getAddCalls()
	assert.Equal(t, []kvblock.BlockHash{9001, 9002}, calls[0].engineKeys)
	assert.Equal(t, []kvblock.BlockHash{100, 200}, calls[0].requestKeys)
	assert.Equal(t, "default/pod-0", calls[0].entries[0].PodIdentifier)
	assert.Equal(t, "gpu", calls[0].entries[0].DeviceTier)

	cancel()
	pool.Shutdown(ctx)
}

func TestPool_EndToEnd_Evict(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{}

	adapter := &mockAdapter{
		parseFunc: func(msg *RawMessage) (string, string, EventBatch, error) {
			return "default/pod-0", "llama-3", EventBatch{
				Events: []GenericEvent{
					&BlockRemovedEvent{
						BlockHashes: []uint64{9001},
						DeviceTier:  "GPU",
					},
				},
			}, nil
		},
	}

	cfg := &Config{Concurrency: 2}
	pool := NewPool(cfg, idx, tp, adapter)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool.Start(ctx)

	pool.AddTask(&RawMessage{
		Topic:   "kv@default/pod-0@llama-3",
		Payload: []byte("test"),
	})

	require.Eventually(t, func() bool {
		return len(idx.getEvictCalls()) == 1
	}, 2*time.Second, 10*time.Millisecond)

	calls := idx.getEvictCalls()
	assert.Equal(t, kvblock.BlockHash(9001), calls[0].key)
	assert.Equal(t, kvblock.EngineKey, calls[0].keyType)
	assert.Equal(t, "default/pod-0", calls[0].entries[0].PodIdentifier)

	cancel()
	pool.Shutdown(ctx)
}

func TestPool_EndToEnd_ParseError(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{}

	adapter := &mockAdapter{
		parseFunc: func(msg *RawMessage) (string, string, EventBatch, error) {
			return "", "", EventBatch{}, fmt.Errorf("parse failed")
		},
	}

	cfg := &Config{Concurrency: 1}
	pool := NewPool(cfg, idx, tp, adapter)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool.Start(ctx)

	pool.AddTask(&RawMessage{Topic: "bad", Payload: []byte("bad")})

	// Wait a bit; no index calls should be made
	time.Sleep(100 * time.Millisecond)

	assert.Empty(t, idx.getAddCalls())
	assert.Empty(t, idx.getEvictCalls())

	cancel()
	pool.Shutdown(ctx)
}

func TestPool_ConcurrentAddTask(t *testing.T) {
	idx := newMockIndex()
	tp := &mockTokenProcessor{returnKeys: []kvblock.BlockHash{100}}

	adapter := &mockAdapter{
		parseFunc: func(msg *RawMessage) (string, string, EventBatch, error) {
			return "default/pod-0", "llama-3", EventBatch{
				Events: []GenericEvent{
					&BlockStoredEvent{
						BlockHashes: []uint64{9001},
						Tokens:      []uint32{1, 2},
					},
				},
			}, nil
		},
	}

	cfg := &Config{Concurrency: 4}
	pool := NewPool(cfg, idx, tp, adapter)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool.Start(ctx)

	var wg sync.WaitGroup
	numTasks := 50
	wg.Add(numTasks)
	for i := 0; i < numTasks; i++ {
		go func(i int) {
			defer wg.Done()
			pool.AddTask(&RawMessage{
				Topic:   fmt.Sprintf("pod-%d", i%5),
				Payload: []byte("data"),
			})
		}(i)
	}
	wg.Wait()

	require.Eventually(t, func() bool {
		return len(idx.getAddCalls()) == numTasks
	}, 5*time.Second, 50*time.Millisecond)

	cancel()
	pool.Shutdown(ctx)
}

// captureModelTokenProcessor captures the parentKey and modelName passed to TokensToKVBlockKeys.
type captureModelTokenProcessor struct {
	returnKeys    []kvblock.BlockHash
	lastParentKey kvblock.BlockHash
	lastModelName string
}

func (c *captureModelTokenProcessor) TokensToKVBlockKeys(
	parentKey kvblock.BlockHash, _ []uint32, modelName string,
) []kvblock.BlockHash {
	c.lastParentKey = parentKey
	c.lastModelName = modelName
	return c.returnKeys
}
