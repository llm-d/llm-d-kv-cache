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

package preprocessing_test

import (
	"context"
	"fmt"
	"os"
	"sync"
	"testing"
	"time"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Global singleton wrapper to prevent multiple Python interpreter initializations.
var (
	globalWrapper     *preprocessing.ChatTemplatingProcessor
	globalWrapperOnce sync.Once
)

// getGlobalWrapper returns a singleton wrapper instance.
func getGlobalWrapper() *preprocessing.ChatTemplatingProcessor {
	globalWrapperOnce.Do(func() {
		globalWrapper = preprocessing.NewChatTemplatingProcessor()
		err := globalWrapper.Initialize()
		if err != nil {
			panic(fmt.Sprintf("Failed to initialize global wrapper: %v", err))
		}
	})
	return globalWrapper
}

// TestGetOrCreateTokenizerKey tests the get_or_create_tokenizer_key function.
func TestGetOrCreateTokenizerKey(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	tests := []struct {
		name           string
		modelName      string
		revision       string
		token          string
		expectTemplate bool
	}{
		{
			name:           "IBM Granite Model",
			modelName:      "ibm-granite/granite-3.3-8b-instruct",
			expectTemplate: true,
		},
		{
			name:           "DialoGPT Model",
			modelName:      "microsoft/DialoGPT-medium",
			expectTemplate: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &preprocessing.GetOrCreateTokenizerKeyRequest{
				Model:    tt.modelName,
				Revision: tt.revision,
				Token:    tt.token,
			}

			// Profile the function call
			start := time.Now()
			_, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
			duration := time.Since(start)

			// Log performance
			t.Logf("Model: %s, Duration: %v", tt.modelName, duration)
			if tt.expectTemplate {
				// Models that should have templates
				require.NoError(t, err, "GetOrCreateTokenizerKey should not return an error")
			} else {
				// Models that don't have chat templates
				if err != nil {
					t.Logf("Expected error for model without chat template: %v", err)
				} else {
					// Some models might return empty template instead of error
					t.Logf("Model returned empty template (expected for non-chat models)")
				}
			}
		})
	}
}

// containsSubsequence checks if all elements of sub appear in seq in the same order (not necessarily contiguous).
func containsSubsequence(seq, sub []uint32) bool {
	if len(sub) == 0 {
		return true
	}
	subIdx := 0
	for _, val := range seq {
		if val == sub[subIdx] {
			subIdx++
			if subIdx == len(sub) {
				return true
			}
		}
	}
	return false
}

// TestRenderChat tests the RenderChat function with both custom and model default templates.
func TestRenderChat(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	// Simple template for testing
	simpleTemplate := `{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}`

	// Complex template for testing
	complexTemplate := `{%- if messages[0]['role'] == 'system' %}
     {%- set system_message = messages[0]['content'] %}
     {%- set loop_messages = messages[1:] %}
 {%- else %}
     {%- set system_message = "You are a helpful assistant." %}
     {%- set loop_messages = messages %}
 {%- endif %}
{{ system_message }}
{%- for message in loop_messages %}
{{ message.role }}: {{ message.content }}
{%- endfor %}`

	tests := []struct {
		name           string
		modelName      string
		template       string // empty string means use model's default template
		messages       []preprocessing.Conversation
		expectedTokens []uint32
	}{
		// Custom template tests
		{
			name:      "Custom Simple Template",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			template:  simpleTemplate,
			messages: []preprocessing.Conversation{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
			},
			expectedTokens: []uint32{0x1f0, 0x2c, 0x2ee0, 0xcb, 0x44ba, 0x2c, 0x1a7a, 0x7e1, 0x13, 0xcb},
		},
		{
			name:      "Custom Complex Template with System Message",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			template:  complexTemplate,
			messages: []preprocessing.Conversation{
				{Role: "system", Content: "You are a helpful AI assistant."},
				{Role: "user", Content: "What is the weather like?"},
				{Role: "assistant", Content: "I don't have access to real-time weather data."},
			},
			expectedTokens: []uint32{
				0x10ba, 0x374, 0x138, 0x435f, 0x4c5f, 0xb8e2, 0x20, 0x1f0, 0x2c, 0x1824,
				0x1b6, 0x142, 0x4e37, 0x84c, 0x31, 0x44ba, 0x2c, 0x1b7, 0xaf0, 0x532,
				0x487, 0xb29, 0x174, 0xfab, 0x1f, 0x3eb, 0x4e37, 0x2c2, 0x20,
			},
		},
		{
			name:      "Custom Complex Template without System Message",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			template:  complexTemplate,
			messages: []preprocessing.Conversation{
				{Role: "user", Content: "Tell me a joke"},
				{Role: "assistant", Content: "Why don't scientists trust atoms? Because they make up everything!"},
			},
			expectedTokens: []uint32{
				0x10ba, 0x374, 0x138, 0x435f, 0xb8e2, 0x20, 0x1f0, 0x2c, 0x788e, 0x255,
				0x138, 0x2215, 0x1df, 0x44ba, 0x2c, 0x3f6f, 0xaf0, 0x532, 0x895, 0x646,
				0xbe5, 0x469a, 0x6cfa, 0x31, 0x47c1, 0xb89, 0x78a, 0x3cd, 0x231d, 0x13,
			},
		},
		// Model default template tests
		{
			name:      "Model Default Template - IBM Granite",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			template:  "", // use model's default template
			messages: []preprocessing.Conversation{
				{Role: "user", Content: "What is the capital of France?"},
				{Role: "assistant", Content: "The capital of France is Paris."},
			},
			// Date tokens removed since IBM Granite includes dynamic date in system prompt
			expectedTokens: []uint32{
				0xc000, 0xb82, 0xc001, 0x9a86, 0x186, 0x42af, 0xb05, 0x2c, 0x7704, 0xe1,
				0x24, 0x22, 0x24, 0x26, 0x20, 0xcb, 0x5c75, 0x49e, 0xb05, 0x2c, 0x20,
				0xcb, 0x10ba, 0x374, 0x1f90, 0x116, 0x293, 0x1e, 0x49dd, 0x32a, 0x6461,
				0x20, 0x990, 0x374, 0x138, 0x435f, 0x4c5f, 0xb8e2, 0x20, 0x0, 0xcb,
				0xc000, 0x1f0, 0xc001, 0x2005, 0x1b6, 0x142, 0x49ee, 0x1b0, 0xb220, 0x31,
				0x0, 0xcb, 0xc000, 0x44ba, 0xc001, 0x526, 0x49ee, 0x1b0, 0xb220,
				0x1b6, 0xa9c, 0x129, 0x20, 0x0, 0xcb,
			},
		},
		{
			name:      "Model Default Template - DialoGPT Multi-turn",
			modelName: "microsoft/DialoGPT-medium",
			template:  "", // use model's default template
			messages: []preprocessing.Conversation{
				{Role: "user", Content: "Hello, how are you?"},
				{Role: "assistant", Content: "I'm doing well, thank you!"},
			},
			expectedTokens: []uint32{
				0x3c88, 0xb, 0x2bf, 0x185, 0x159, 0x1e, 0xc450, 0x28, 0x44d, 0x70c,
				0x370, 0xb, 0x16f3, 0x159, 0x0, 0xc450,
			},
		},
		{
			name:      "Model Default Template - System Message",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			template:  "", // use model's default template
			messages: []preprocessing.Conversation{
				{Role: "system", Content: "You are a helpful AI assistant specialized in coding."},
				{Role: "user", Content: "Write a Python function to calculate fibonacci numbers."},
				{
					Role: "assistant",
					Content: "Here's a Python function:\ndef fibonacci(n):\n" +
						"    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
				},
			},
			expectedTokens: []uint32{
				0xc000, 0xb82, 0xc001, 0x10ba, 0x374, 0x138, 0x435f, 0x4c5f, 0xb8e2,
				0xaef5, 0x148, 0x2975, 0x20, 0x0, 0xcb, 0xc000, 0x1f0, 0xc001, 0x9ea,
				0x138, 0x1301, 0x29b, 0x174, 0x23d1, 0x6e10, 0x877a, 0x1d5b, 0x20,
				0x0, 0xcb, 0xc000, 0x44ba, 0xc001, 0x2aa9, 0x49e, 0x138, 0x1301,
				0x29b, 0x2c, 0xcb, 0x24d, 0x6e10, 0x877a, 0x1a, 0x60, 0x2c7, 0x11c,
				0x1ba, 0x136, 0x19f, 0x136, 0x9cf, 0xe1, 0x23, 0x32d, 0x6e10, 0x877a,
				0x1a, 0x60, 0x1f, 0x23, 0x1b, 0x1da, 0x6e10, 0x877a, 0x1a, 0x60,
				0x1f, 0x24, 0x1b, 0x0, 0xcb,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			key, err := wrapper.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
				Model:   tt.modelName,
				IsLocal: true,
			})
			require.NoError(t, err, "Failed to get tokenizer key")

			start := time.Now()
			tokens, _, err := wrapper.RenderChat(ctx, &preprocessing.RenderChatRequest{
				Key:          key,
				Conversation: tt.messages,
				ChatTemplate: tt.template,
			})
			duration := time.Since(start)

			require.NoError(t, err, "RenderChat should not return an error")
			assert.NotEmpty(t, tokens, "tokens should not be empty")

			t.Logf("Test: %s, Model: %s, Duration: %v, Token count: %d", tt.name, tt.modelName, duration, len(tokens))

			assert.True(t, containsSubsequence(tokens, tt.expectedTokens),
				"Actual tokens should contain expected tokens as subsequence")
		})
	}
}

// TestRender tests the render function.
func TestRender(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	tests := []struct {
		name           string
		modelName      string
		revision       string
		hfToken        string
		expectedTokens []uint32
	}{
		{
			name:           "IBM Granite Model",
			modelName:      "ibm-granite/granite-3.3-8b-instruct",
			expectedTokens: []uint32{0x2057, 0x1e, 0xa40, 0x374, 0x34c, 0x31},
		},
		{
			name:           "DialoGPT Model",
			modelName:      "microsoft/DialoGPT-medium",
			expectedTokens: []uint32{0x3c88, 0xb, 0x2bf, 0x185, 0x159, 0x1e},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			key, err := wrapper.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
				Model:   tt.modelName,
				IsLocal: true,
			})
			require.NoError(t, err, "Failed to get tokenizer key")

			request := &preprocessing.RenderRequest{
				Key:              key,
				Text:             "Hello, how are you?",
				AddSpecialTokens: true,
			}

			// Profile the function call
			start := time.Now()
			tokens, offsets, err := wrapper.Render(context.Background(), request)
			duration := time.Since(start)

			// Log performance
			t.Logf("Model: %s, Duration: %v, Tokens length: %d", tt.modelName, duration, len(tokens))

			// Models that should have templates
			require.NoError(t, err, "Render should not return an error")
			assert.NotEmpty(t, tokens, "Tokens should not be empty")
			assert.NotNil(t, offsets, "Offsets should not be nil")
			assert.Equal(t, tt.expectedTokens, tokens, "Rendered tokens should match expected tokens")
		})
	}
}

// TestGetOrCreateTokenizerKeyCaching tests the caching functionality.
func TestGetOrCreateTokenizerKeyCaching(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear all caches to ensure we start with a clean state
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	modelName := "ibm-granite/granite-3.3-8b-instruct"
	request := &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   modelName,
		IsLocal: true,
	}

	// First call - should be cache miss
	t.Log("=== First call (Cache MISS) ===")
	start := time.Now()
	key1, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
	duration1 := time.Since(start)
	require.NoError(t, err, "First call should not return an error")

	// Second call - should be cache hit
	t.Log("=== Second call (Cache HIT) ===")
	start = time.Now()
	key2, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
	duration2 := time.Since(start)
	require.NoError(t, err, "Second call should not return an error")

	// Verify that both calls returned the same key
	assert.Equal(t, key1, key2, "Both calls should return the same tokenizer key")

	// Verify performance improvement
	t.Logf("First call duration: %v, Second call duration: %v, Speedup: %.1fx",
		duration1, duration2, float64(duration1)/float64(duration2))

	// Cache hit should be significantly faster
	assert.Less(t, duration2, duration1, "Cache hit should be faster than cache miss")
}

// TestRenderChatWithDocuments tests RenderChat with Documents and ChatTemplateKWArgs fields.
func TestRenderChatWithDocuments(t *testing.T) {
	wrapper := getGlobalWrapper()
	ctx := context.Background()

	tests := []struct {
		name           string
		modelName      string
		expectedTokens []uint32
	}{
		{
			name:      "IBM Granite with Documents",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			// Date tokens removed since IBM Granite includes dynamic date in system prompt
			expectedTokens: []uint32{
				0xc000, 0xb82, 0xc001, 0x9a86, 0x186, 0x42af, 0xb05, 0x2c, 0x7704,
				0xe1, 0x24, 0x22, 0x24, 0x26, 0x20, 0xcb, 0x5c75, 0x49e, 0xb05,
				0x2c, 0x20, 0xcb, 0x10ba, 0x374, 0x1f90, 0x116, 0x293, 0x1e,
				0x49dd, 0x32a, 0x6461, 0x20, 0x173e, 0x142, 0x6fd, 0x174, 0x142,
				0x4e8, 0x49e, 0x5e5, 0x32a, 0x81c4, 0xbc6, 0x12b, 0x26f, 0x142,
				0xac14, 0x148, 0x142, 0xf67, 0x321b, 0x20, 0x686, 0x142, 0x9a7,
				0x14e5, 0x174, 0x1da8, 0x142, 0x1b58, 0x1b6, 0x286, 0xce8, 0x148,
				0x142, 0x321b, 0x1e, 0x1f14, 0x142, 0x4e8, 0x2b0, 0x142, 0x1b58,
				0x1311, 0x20e, 0x9c5f, 0x101a, 0x220, 0x142, 0xce8, 0x2c2, 0x20,
				0x0, 0xcb, 0xc000, 0xafc, 0xd77, 0xafc, 0x51, 0x13a, 0x233,
				0xd02, 0x6f, 0xc001, 0xcb, 0x526, 0x4e37, 0x148, 0xa9c, 0x129,
				0x1b6, 0x3bdb, 0x15ba, 0x1cd, 0xe1, 0x24, 0x27, 0x5cd7, 0x35,
				0x20, 0x0, 0xcb, 0xc000, 0x1f0, 0xc001, 0x2005, 0x1b6, 0x142,
				0x4e37, 0x148, 0xa9c, 0x129, 0x31, 0x0, 0xcb, 0xc000, 0x44ba,
				0xc001, 0x2651, 0x255, 0x5e1, 0x2b0, 0x1b4, 0x34c, 0x20, 0x0,
				0xcb,
			},
		},
		{
			name:      "TinyLlama with Documents",
			modelName: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			expectedTokens: []uint32{
				0x211, 0x7525, 0x700, 0x7525, 0x7506, 0xd, 0x15f2, 0x152, 0x116,
				0x39ea, 0x129, 0xe61, 0x7515, 0x2, 0x74af, 0xd, 0x750e, 0x7525,
				0x1d1, 0x5679, 0x7525, 0x7506, 0xd, 0x2ef8, 0x250, 0x58f, 0x189,
				0x16b, 0x16e, 0x74c1, 0x2, 0x74af, 0xd,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key, err := wrapper.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
				Model:   tt.modelName,
				IsLocal: true,
			})
			require.NoError(t, err, "Failed to get tokenizer key")

			tokens, _, err := wrapper.RenderChat(ctx, &preprocessing.RenderChatRequest{
				Key: key,
				Conversation: []preprocessing.Conversation{
					{Role: "user", Content: "What is the weather in Paris?"},
					{Role: "assistant", Content: "Let me check that for you."},
				},
				Documents: []interface{}{
					map[string]interface{}{
						"title": "Paris Weather Report",
						"text":  "The weather in Paris is sunny and 25°C.",
					},
				},
				ChatTemplate: "",
				ChatTemplateKWArgs: map[string]interface{}{
					"max_tokens":  10,
					"temperature": 0.0,
				},
			})
			require.NoError(t, err, "Failed to render chat template")
			assert.True(t, containsSubsequence(tokens, tt.expectedTokens),
				"Actual tokens should contain expected tokens as subsequence")
		})
	}
}

// TestLongChatCompletions tests with longer, more complex conversations.
func TestLongChatCompletions(t *testing.T) {
	wrapper := getGlobalWrapper()

	ctx := context.Background()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(ctx)
	require.NoError(t, err, "Failed to clear caches")

	// Create a long conversation
	longConversation := []preprocessing.Conversation{
		{Role: "system", Content: "You are an expert software engineer with deep knowledge of Go, Python, " +
			"and system design. " +
			"Provide detailed, accurate responses."},
		{Role: "user", Content: "I'm building a high-performance caching system in Go. Can you help me design " +
			"the architecture?"},
		{Role: "assistant", Content: "Absolutely! For a high-performance caching system in Go, I'd recommend " +
			"starting with a layered architecture. Let's break this down into components."},
		{Role: "user", Content: "What about memory management and eviction policies?"},
		{Role: "assistant", Content: "Great question! Memory management is crucial. I'd suggest implementing an " +
			"LRU (Least Recently Used) eviction policy " +
			"with configurable memory limits. You can use a combination of a hash map for O(1) lookups and a " +
			"doubly-linked list for tracking access order."},
		{Role: "user", Content: "How should I handle concurrent access and thread safety?"},
		{Role: "assistant", Content: "For thread safety, you have several options. The most common approach is " +
			"to use sync.RWMutex for read-write locks, " +
			"allowing multiple concurrent readers but exclusive writers. Alternatively, you could use sync.Map " +
			"for simpler cases or implement a lock-free design " +
			"with atomic operations for maximum performance."},
		{Role: "user", Content: "What about persistence and recovery?"},
		{Role: "assistant", Content: "For persistence, consider using a write-ahead log (WAL) pattern. This " +
			"involves logging all mutations to disk before applying them to memory. " +
			"For recovery, you can replay the log to reconstruct the cache state. You might also want to " +
			"implement periodic snapshots for faster recovery."},
	}

	modelName := "ibm-granite/granite-3.3-8b-instruct"

	t.Run("Long Conversation Processing", func(t *testing.T) {
		// Render long conversation
		start := time.Now()
		key, err := wrapper.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
			Model:   modelName,
			IsLocal: true,
		})
		require.NoError(t, err, "Failed to get tokenizer key")
		tokens, _, err := wrapper.RenderChat(ctx, &preprocessing.RenderChatRequest{
			Key:          key,
			Conversation: longConversation,
		})
		renderDuration := time.Since(start)
		require.NoError(t, err, "Failed to render long conversation")

		// Verify results
		assert.NotEmpty(t, tokens, "Long conversation should render successfully")
		assert.Greater(t, len(tokens), 300,
			"Long conversation should produce substantial output")

		// Performance metrics
		t.Logf("ChatTemplate Long conversation render: %v", renderDuration)
	})
}

// BenchmarkGetOrCreateTokenizerKey benchmarks the template fetching performance.
func BenchmarkGetOrCreateTokenizerKey(b *testing.B) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(b, err, "Failed to clear caches")

	request := &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model: "ibm-granite/granite-3.3-8b-instruct",
	}

	// Track first iteration time and total time
	var firstIterationTime time.Duration
	var totalTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		_, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
		require.NoError(b, err, "Benchmark should not return errors")
		iterTime := time.Since(start)

		if i == 0 {
			firstIterationTime = iterTime
		}
		totalTime += iterTime
	}

	// Calculate both overall average and warm performance average
	overallAvg := totalTime / time.Duration(b.N)

	var warmAvg time.Duration
	if b.N > 1 {
		warmAvg = (totalTime - firstIterationTime) / time.Duration(b.N-1)
	} else {
		warmAvg = overallAvg // If only one iteration, warm avg = overall avg
	}

	b.ReportMetric(float64(overallAvg.Nanoseconds()), "ns/op_overall")
	b.ReportMetric(float64(warmAvg.Nanoseconds()), "ns/op_warm")
}

// BenchmarkRenderChat benchmarks the chat rendering performance.
func BenchmarkRenderChat(b *testing.B) {
	wrapper := getGlobalWrapper()

	ctx := context.Background()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(ctx)
	require.NoError(b, err, "Failed to clear caches")

	key, err := wrapper.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   "ibm-granite/granite-3.3-8b-instruct",
		IsLocal: true,
	})
	require.NoError(b, err, "Failed to get tokenizer key")

	// Track first iteration time and total time
	var firstIterationTime time.Duration
	var totalTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		_, _, err := wrapper.RenderChat(ctx, &preprocessing.RenderChatRequest{
			Key: key,
			Conversation: []preprocessing.Conversation{
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "Hi there!"},
			},
		})
		require.NoError(b, err, "Benchmark should not return errors")
		iterTime := time.Since(start)

		if i == 0 {
			firstIterationTime = iterTime
		}
		totalTime += iterTime
	}

	// Calculate both overall average and warm performance average
	overallAvg := totalTime / time.Duration(b.N)

	var warmAvg time.Duration
	if b.N > 1 {
		warmAvg = (totalTime - firstIterationTime) / time.Duration(b.N-1)
	} else {
		warmAvg = overallAvg // If only one iteration, warm avg = overall avg
	}

	b.ReportMetric(float64(overallAvg.Nanoseconds()), "ns/op_overall")
	b.ReportMetric(float64(warmAvg.Nanoseconds()), "ns/op_warm")
}

// BenchmarkRender benchmarks the render performance.
func BenchmarkRender(b *testing.B) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(b, err, "Failed to clear caches")

	ctx := context.Background()
	key, err := wrapper.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   "ibm-granite/granite-3.3-8b-instruct",
		IsLocal: true,
	})
	require.NoError(b, err, "Failed to get tokenizer key")

	// Track first iteration time and total time
	var firstIterationTime time.Duration
	var totalTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		_, _, err := wrapper.Render(ctx, &preprocessing.RenderRequest{
			Key:              key,
			Text:             "What is the capital of France?",
			AddSpecialTokens: true,
		})
		require.NoError(b, err, "Benchmark should not return errors")
		iterTime := time.Since(start)

		if i == 0 {
			firstIterationTime = iterTime
		}
		totalTime += iterTime
	}

	// Calculate both overall average and warm performance average
	overallAvg := totalTime / time.Duration(b.N)

	var warmAvg time.Duration
	if b.N > 1 {
		warmAvg = (totalTime - firstIterationTime) / time.Duration(b.N-1)
	} else {
		warmAvg = overallAvg // If only one iteration, warm avg = overall avg
	}

	b.ReportMetric(float64(overallAvg.Nanoseconds()), "ns/op_overall")
	b.ReportMetric(float64(warmAvg.Nanoseconds()), "ns/op_warm")
}

// TestLocalTokenizer tests local tokenizer functionality including key creation, RenderChat, and Render.
func TestLocalTokenizer(t *testing.T) {
	wrapper := getGlobalWrapper()
	testModelPath := "../../tokenization/testdata/test-model"

	// Get tokenizer key once for all subtests
	key, err := wrapper.GetOrCreateTokenizerKey(context.Background(), &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   testModelPath,
		IsLocal: true,
	})
	require.NoError(t, err, "GetOrCreateTokenizerKey should not return an error for local path")
	assert.NotEmpty(t, key, "Returned tokenizer key should not be empty")

	t.Run("RenderChat", func(t *testing.T) {
		tokens, offset, err := wrapper.RenderChat(context.Background(), &preprocessing.RenderChatRequest{
			Key: key,
			Conversation: []preprocessing.Conversation{
				{Role: "user", Content: "Hello from local tokenizer!"},
				{Role: "assistant", Content: "Hi! I'm using a locally loaded template."},
			},
		})
		require.NoError(t, err, "RenderChat should not return an error")
		assert.NotEmpty(t, tokens, "tokens should not be empty")
		assert.NotNil(t, offset, "offset should not be nil")
		assert.Contains(t, tokens, uint32(7592), "tokens should contain 7592(hello)")
	})

	t.Run("Render", func(t *testing.T) {
		tokens, offset, err := wrapper.Render(context.Background(), &preprocessing.RenderRequest{
			Key:              key,
			Text:             "Hello from local tokenizer!",
			AddSpecialTokens: true,
		})
		require.NoError(t, err, "Render should not return an error for local path")
		assert.NotEmpty(t, tokens, "tokens should not be empty")
		assert.NotNil(t, offset, "offset should not be nil")
		assert.Contains(t, tokens, uint32(7592), "tokens should contain 7592(hello)")
	})
}

// TestGetOrCreateTokenizerKeyLocalPathCaching tests that local templates are cached properly.
func TestGetOrCreateTokenizerKeyLocalPathCaching(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches first
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	testModelPath := "../../tokenization/testdata/test-model"
	request := &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   testModelPath,
		IsLocal: true,
	}

	// First call - cache miss
	start := time.Now()
	key1, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
	duration1 := time.Since(start)
	require.NoError(t, err, "First call should not return an error")

	// Second call - cache hit
	start = time.Now()
	key2, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
	duration2 := time.Since(start)
	require.NoError(t, err, "Second call should not return an error")

	// Verify that both calls returned the same key
	assert.Equal(t, key1, key2, "Both calls should return the same tokenizer key")

	// Cache hit should be faster
	t.Logf("First call (cache miss): %v, Second call (cache hit): %v, Speedup: %.1fx",
		duration1, duration2, float64(duration1)/float64(duration2))
	assert.Less(t, duration2, duration1, "Cache hit should be faster than cache miss")
}

// TestGetOrCreateTokenizerKeyLocalPathWithFile tests loading from a specific tokenizer.json file path.
func TestGetOrCreateTokenizerKeyLocalPathWithFile(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Test with the full path to tokenizer.json
	//nolint:gosec // This is a test file path, not a credential
	testTokenizerPath := "../../tokenization/testdata/test-model/tokenizer.json"

	request := &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   testTokenizerPath,
		IsLocal: true,
	}

	// Get the tokenizer key
	key, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
	require.NoError(t, err, "GetOrCreateTokenizerKey should handle file path and extract directory")
	assert.NotEmpty(t, key, "Returned tokenizer key should not be empty")

	t.Logf("Loaded tokenizer from file path: %s", testTokenizerPath)
}

// TestGetOrCreateTokenizerKeyLocalPathNonExistent tests error handling for non-existent local paths.
func TestGetOrCreateTokenizerKeyLocalPathNonExistent(t *testing.T) {
	wrapper := getGlobalWrapper()

	request := &preprocessing.GetOrCreateTokenizerKeyRequest{
		Model:   "/non/existent/path",
		IsLocal: true,
	}

	// This should return an error
	key, err := wrapper.GetOrCreateTokenizerKey(context.Background(), request)
	// Assertions
	assert.Error(t, err, "GetOrCreateTokenizerKey should return an error for non-existent path")
	assert.Empty(t, key, "Returned tokenizer key should be empty for non-existent path")
	t.Logf("Expected error for non-existent path: %v", err)
}

// TestMain provides a controlled setup and teardown for tests in this package.
func TestMain(m *testing.M) {
	// Create a new processor to handle initialization.
	processor := preprocessing.NewChatTemplatingProcessor()

	// Set up: Initialize the Python interpreter.
	log.Log.Info("Initializing Python interpreter for tests...")
	if err := processor.Initialize(); err != nil {
		log.Log.Error(err, "Failed to initialize Python interpreter")
		os.Exit(1)
	}
	log.Log.Info("Python interpreter initialized successfully.")

	// Run all the tests in the package.
	exitCode := m.Run()

	// Tear down: Finalize the Python interpreter.
	log.Log.Info("Finalizing Python interpreter...")
	processor.Finalize()
	log.Log.Info("Python interpreter finalized.")

	// Exit with the result of the test run.
	os.Exit(exitCode)
}
