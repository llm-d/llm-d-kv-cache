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

//nolint:testpackage // need to test internal types
package tokenization

import (
	"context"
	"math/rand"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"k8s.io/client-go/util/workqueue"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
)

const (
	benchmarkMaxWords    = 1_000
	benchmarkWordLength  = 2
	benchmarkSeed        = 42
	benchmarkWorkerCount = 5
)

var benchmarkModels = []string{
	"google-bert/bert-base-uncased",
	"openai-community/gpt2",
}

// MockTokenizer implements the Tokenizer interface for testing.
type MockTokenizer struct {
	mock.Mock
}

func (m *MockTokenizer) RenderChat(renderReq *preprocessing.RenderChatRequest) ([]uint32, []preprocessing.Offset, error) {
	args := m.Called(renderReq)
	tokenIface := args.Get(0)
	if tokenIface == nil {
		return nil, nil, args.Error(2)
	}
	tokens, ok := tokenIface.([]uint32)
	if !ok {
		panic("MockTokenizer.RenderChat: expected []uint32 from mock, got unexpected type")
	}
	offsetIface := args.Get(1)
	if offsetIface == nil {
		return nil, nil, args.Error(2)
	}
	offsets, ok := offsetIface.([]preprocessing.Offset)
	if !ok {
		panic("MockTokenizer.RenderChat: expected []preprocessing.Offset from mock, got unexpected type")
	}
	return tokens, offsets, args.Error(2)
}

func (m *MockTokenizer) Render(prompt string) ([]uint32, []preprocessing.Offset, error) {
	args := m.Called(prompt)
	tokenIface := args.Get(0)
	if tokenIface == nil {
		return nil, nil, args.Error(2)
	}
	tokens, ok := tokenIface.([]uint32)
	if !ok {
		panic("MockTokenizer.Render: expected []uint32 from mock, got unexpected type")
	}
	offsetIface := args.Get(1)
	if offsetIface == nil {
		return nil, nil, args.Error(2)
	}
	offsets, ok := offsetIface.([]preprocessing.Offset)
	if !ok {
		panic("MockTokenizer.Render: expected []preprocessing.Offset from mock, got unexpected type")
	}
	return tokens, offsets, args.Error(2)
}

func (m *MockTokenizer) Close() error {
	return nil
}

func (m *MockTokenizer) Type() string {
	return "mock"
}

// MockIndexer implements the prefixstore.Indexer interface for testing.
type MockIndexer struct {
	mock.Mock
}

func TestPool_ProcessTask(t *testing.T) {
	mockIndexer := &MockIndexer{}
	mockTokenizer := &MockTokenizer{}

	pool := &Pool{
		modelName: testModelName,
		workers:   1,
		tokenizer: mockTokenizer,
	}

	task := Task{
		Prompt: "hello world",
	}

	// Setup specific mock return values
	expectedTokens := []uint32{12345, 67890, 11111}
	expectedOffsets := []preprocessing.Offset{{0, 5}, {6, 11}}

	mockTokenizer.On("Render", task.Prompt).
		Return(expectedTokens, expectedOffsets, nil)

	// Execute
	err := pool.processTask(task)

	// Assert
	assert.NoError(t, err)
	mockTokenizer.AssertExpectations(t)
	mockIndexer.AssertExpectations(t)
}

func TestPool_WorkerLoop(t *testing.T) {
	specs := map[string]struct {
		setupMocks func(*MockIndexer, *MockTokenizer)
		genTasks   func() ([]Task, chan tokenizationResponse)
		verify     func(t *testing.T, pool *Pool, tasks []Task, resultChan chan tokenizationResponse)
	}{
		"successful task processing": {
			setupMocks: func(mi *MockIndexer, mt *MockTokenizer) {
				mt.On("Render", "test prompt").
					Return([]uint32{1, 2, 3}, []preprocessing.Offset{{0, 4}}, nil)
			},
			genTasks: func() ([]Task, chan tokenizationResponse) {
				return []Task{{Prompt: "test prompt"}}, nil
			},
			verify: func(t *testing.T, pool *Pool, tasks []Task, resultChan chan tokenizationResponse) {}, //nolint:thelper // noop
		},
		"task with result channel": {
			setupMocks: func(mi *MockIndexer, mt *MockTokenizer) {
				mt.On("Render", "test with channel").
					Return([]uint32{10, 20, 30}, []preprocessing.Offset{{0, 4}}, nil)
			},
			genTasks: func() ([]Task, chan tokenizationResponse) {
				ch := make(chan tokenizationResponse, 1)
				return []Task{{
					Prompt:   "test with channel",
					ResultCh: ch,
				}}, ch
			},
			verify: func(t *testing.T, pool *Pool, tasks []Task, resultCh chan tokenizationResponse) {
				t.Helper()
				require.Eventually(t, func() bool {
					if result, ok := <-resultCh; ok {
						assert.Equal(t, []uint32{10, 20, 30}, result.Tokens)
						return true
					}
					return false
				}, time.Second, 10*time.Millisecond)

				// Verify channel is closed
				require.Eventually(t, func() bool {
					_, ok := <-resultCh
					return !ok
				}, time.Second, 10*time.Millisecond)
			},
		},
		"multiple tasks processing": {
			setupMocks: func(mi *MockIndexer, mt *MockTokenizer) {
				for i := range 5 {
					prompt := "prompt " + string(rune('a'+i))
					tokens := []uint32{uint32(i), uint32(i + 1)} //nolint:gosec // test code
					offsets := []preprocessing.Offset{{0, 6}}

					mt.On("Render", prompt).
						Return(tokens, offsets, nil).Once()
				}
			},
			genTasks: func() ([]Task, chan tokenizationResponse) {
				tasks := make([]Task, 5)
				for i := range 5 {
					tasks[i] = Task{Prompt: "prompt " + string(rune('a'+i))}
				}
				return tasks, nil
			},
			verify: func(t *testing.T, pool *Pool, tasks []Task, resultChan chan tokenizationResponse) {
				t.Helper()
				require.Eventually(t, func() bool {
					return pool.queue.Len() == 0
				}, time.Second, 10*time.Millisecond, "queue should be drained")
			},
		},
		"max retries exceeded": {
			setupMocks: func(mi *MockIndexer, mt *MockTokenizer) {
				// Mock will fail every time, causing retries
				mt.On("Render", "failing prompt").Return(
					[]uint32{}, []preprocessing.Offset{}, assert.AnError)
			},
			genTasks: func() ([]Task, chan tokenizationResponse) {
				ch := make(chan tokenizationResponse, 1)
				return []Task{{
					Prompt:   "failing prompt",
					ResultCh: ch,
				}}, ch
			},
			verify: func(t *testing.T, pool *Pool, tasks []Task, resultCh chan tokenizationResponse) {
				t.Helper()
				require.Eventually(t, func() bool { // channel is closed, when max retries exceeded
					if result, ok := <-resultCh; !ok {
						assert.Equal(t, tokenizationResponse{}, result)
						return true
					}
					return false
				}, time.Second, 10*time.Millisecond)

				require.Eventually(t, func() bool {
					return pool.queue.Len() == 0
				}, time.Second, 10*time.Millisecond)
			},
		},
	}
	for name, tt := range specs {
		t.Run(name, func(t *testing.T) {
			mockIndexer := &MockIndexer{}
			mockTokenizer := &MockTokenizer{}

			tt.setupMocks(mockIndexer, mockTokenizer)
			pool := &Pool{
				modelName: testModelName,
				workers:   1,
				queue:     workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[Task]()),
				tokenizer: mockTokenizer,
			}

			tasks, resultChan := tt.genTasks()
			for _, task := range tasks {
				pool.queue.Add(task)
			}

			pool.wg.Add(1)
			go pool.workerLoop(0)

			tt.verify(t, pool, tasks, resultChan)

			// Shutdown
			pool.queue.ShutDown()
			pool.wg.Wait()

			// Assert expectations
			mockTokenizer.AssertExpectations(t)
			mockIndexer.AssertExpectations(t)
		})
	}
}

func TestPool_RunIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping tokenizer integration test in short mode")
	}

	mockIndexer := &MockIndexer{}

	prompts := []string{"hello world", "this is a test", "unicode test: 世界"}

	config := &Config{
		ModelName:         testModelName,
		WorkersCount:      5,
		HFTokenizerConfig: DefaultHFTokenizerConfig(),
	}

	// Create context for the pool
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	pool, err := NewTokenizationPool(ctx, config)
	require.NoError(t, err)

	for _, prompt := range prompts {
		pool.EnqueueTokenization(prompt)
	}

	// Run pool
	done := make(chan struct{})
	go func() {
		defer close(done)
		pool.Run(ctx)
	}()

	time.Sleep(2 * time.Second)
	cancel()
	<-done

	mockIndexer.AssertExpectations(t)
}

func generateRandomSentence(wordLength, maxWords int, rng *rand.Rand) string {
	numWords := rng.Intn(maxWords) + 1
	words := make([]string, numWords)

	for i := range numWords {
		word := make([]byte, wordLength)
		for j := range wordLength {
			word[j] = byte('a' + rng.Intn(26))
		}
		words[i] = string(word)
	}

	return strings.Join(words, " ")
}

func setupStressTest(b *testing.B, modelName string) *Pool {
	b.Helper()

	config := &Config{
		ModelName:         modelName,
		WorkersCount:      benchmarkWorkerCount,
		HFTokenizerConfig: DefaultHFTokenizerConfig(),
	}

	pool, err := NewTokenizationPool(context.Background(), config)
	require.NoError(b, err)
	return pool
}

func BenchmarkAsyncTokenizationStress(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping tokenizer integration test in short mode")
	}

	for _, modelName := range benchmarkModels {
		b.Run(modelName, func(b *testing.B) {
			pool := setupStressTest(b, modelName)

			// Return RNG for on-demand prompt generation
			rng := rand.New(rand.NewSource(benchmarkSeed)) //nolint:gosec // Test code - weak random is acceptable

			// Generate and enqueue prompts on-the-fly to avoid memory bloat
			for range b.N {
				prompt := generateRandomSentence(benchmarkWordLength, benchmarkMaxWords, rng)
				pool.EnqueueTokenization(prompt)
			}

			// Create context for the pool
			ctx, cancel := context.WithCancel(context.Background())

			// Run pool
			go pool.Run(ctx)

			b.ResetTimer()

			// when pool gets empty pool.queue.Len() == 0 call cancel to the context:
			for pool.queue.Len() > 0 {
				time.Sleep(100 * time.Millisecond)
			}

			b.StopTimer()
			cancel()

			frequency := float64(b.N) / b.Elapsed().Seconds()
			b.Logf("%s - Processed %d tasks in %v (%.2f tasks/sec)",
				modelName, b.N, b.Elapsed(), frequency)
		})
	}
}

func BenchmarkSyncTokenizationStress(b *testing.B) {
	if testing.Short() {
		b.Skip("Skipping tokenizer integration test in short mode")
	}

	for _, modelName := range benchmarkModels {
		b.Run(modelName, func(b *testing.B) {
			pool := setupStressTest(b, modelName)

			// Return RNG for on-demand prompt generation
			rng := rand.New(rand.NewSource(benchmarkSeed)) //nolint:gosec // Test code - weak random is acceptable

			// Create context for the pool
			ctx, cancel := context.WithCancel(context.Background())

			// Run pool
			go pool.Run(ctx)

			// Now that workers are running, reset benchmark timer
			b.ResetTimer()

			// Submit tokenization requests in a loop until limit
			for i := 0; b.Loop(); i++ {
				prompt := generateRandomSentence(benchmarkWordLength, benchmarkMaxWords, rng)
				pool.Tokenize(nil, prompt)
			}

			b.StopTimer()
			cancel()

			frequency := float64(b.N) / b.Elapsed().Seconds()
			b.Logf("%s - Processed %d tasks in %v (%.2f tasks/sec)",
				modelName, b.N, b.Elapsed(), frequency)
		})
	}
}
