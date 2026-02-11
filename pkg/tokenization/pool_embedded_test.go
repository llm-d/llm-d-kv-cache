//go:build embedded_tokenizers

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

	"github.com/stretchr/testify/require"
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

func TestPool_RunIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping tokenizer integration test in short mode")
	}

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
