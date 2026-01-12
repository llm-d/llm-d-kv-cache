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
	"strings"
	"sync"
	"testing"
	"time"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type (
	RenderJinjaTemplateRequest = preprocessing.RenderJinjaTemplateRequest
	Message                    = preprocessing.Message
	Content                    = preprocessing.Content
)

// Global singleton wrapper to prevent multiple Python interpreter initializations.
var (
	globalWrapper     *preprocessing.ChatTemplatingProcessor
	globalWrapperOnce sync.Once
	globalWrapperMu   sync.Mutex
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

// TestGetModelChatTemplate tests the get_model_chat_template function.
func TestGetModelChatTemplate(t *testing.T) {
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
			request := preprocessing.FetchChatTemplateRequest{
				Model:    tt.modelName,
				Revision: tt.revision,
				Token:    tt.token,
			}

			// Profile the function call
			start := time.Now()
			template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), request)
			duration := time.Since(start)

			// Log performance
			t.Logf("Model: %s, Duration: %v, ChatTemplate length: %d", tt.modelName, duration, len(template))

			if tt.expectTemplate {
				// Models that should have templates
				require.NoError(t, err, "FetchChatTemplate should not return an error")
				assert.NotEmpty(t, template, "ChatTemplate should not be empty")
				assert.NotNil(t, templateVars, "ChatTemplate vars should not be nil")
				assert.Contains(t, template, "messages", "ChatTemplate should contain messages")
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

// TestRenderJinjaTemplate tests the render_jinja_template function.
func TestRenderJinjaTemplate(t *testing.T) {
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
		name     string
		template string
		messages []Message
	}{
		{
			name:     "Simple ChatTemplate",
			template: simpleTemplate,
			messages: []Message{
				{Role: "user", Content: Content{Raw: "Hello"}},
				{Role: "assistant", Content: Content{Raw: "Hi there!"}},
			},
		},
		{
			name:     "Complex ChatTemplate with System Message",
			template: complexTemplate,
			messages: []Message{
				{Role: "system", Content: Content{Raw: "You are a helpful AI assistant."}},
				{Role: "user", Content: Content{Raw: "What is the weather like?"}},
				{Role: "assistant", Content: Content{Raw: "I don't have access to real-time weather data."}},
			},
		},
		{
			name:     "Complex ChatTemplate without System Message",
			template: complexTemplate,
			messages: []Message{
				{Role: "user", Content: Content{Raw: "Tell me a joke"}},
				{Role: "assistant", Content: Content{Raw: "Why don't scientists trust atoms? Because they make up everything!"}},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := &RenderJinjaTemplateRequest{
				Messages:     tt.messages,
				ChatTemplate: tt.template,
			}

			// Profile the function call
			start := time.Now()
			response, err := wrapper.RenderChatTemplate(context.Background(), request)
			duration := time.Since(start)

			// Assertions
			require.NoError(t, err, "RenderChatTemplate should not return an error")
			assert.NotNil(t, response, "Response should not be nil")
			assert.NotEmpty(t, response.RenderedChats, "Rendered chats should not be empty")

			// Log performance
			t.Logf("ChatTemplate: %s, Duration: %v, Rendered length: %d", tt.name, duration, len(response.RenderedChats[0]))

			// Verify rendered content
			rendered := response.RenderedChats[0]
			for _, message := range tt.messages {
				// For complex templates, the role might not be explicitly shown in output
				// but the content should always be present
				assert.Contains(t, rendered, message.Content.Raw, "Rendered content should contain message content")

				// Only check for role if it's a simple template (not complex with system message)
				if !strings.Contains(tt.name, "Complex") {
					assert.Contains(t, rendered, message.Role, "Rendered content should contain role")
				}
			}
		})
	}
}

// TestTemplateCaching tests the caching functionality.
func TestTemplateCaching(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear all caches to ensure we start with a clean state
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	modelName := "ibm-granite/granite-3.3-8b-instruct"
	request := preprocessing.FetchChatTemplateRequest{
		Model: modelName,
	}

	// First call - should be cache miss
	t.Log("=== First call (Cache MISS) ===")
	start := time.Now()
	template1, vars1, err := wrapper.FetchChatTemplate(context.Background(), request)
	duration1 := time.Since(start)
	require.NoError(t, err, "First call should not return an error")

	// Second call - should be cache hit
	t.Log("=== Second call (Cache HIT) ===")
	start = time.Now()
	template2, vars2, err := wrapper.FetchChatTemplate(context.Background(), request)
	duration2 := time.Since(start)
	require.NoError(t, err, "Second call should not return an error")

	// Verify results are identical
	assert.Equal(t, template1, template2, "Cached and non-cached results should be identical")
	assert.Equal(t, vars1, vars2, "Cached and non-cached vars should be identical")

	// Verify performance improvement
	t.Logf("First call duration: %v, Second call duration: %v, Speedup: %.1fx",
		duration1, duration2, float64(duration1)/float64(duration2))

	// Cache hit should be significantly faster
	assert.Less(t, duration2, duration1, "Cache hit should be faster than cache miss")
}

// TestChatCompletionsIntegration tests the complete chat completions workflow.
func TestChatCompletionsIntegration(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	tests := []struct {
		name         string
		modelName    string
		conversation []Message
		description  string
	}{
		{
			name:      "Simple Conversation",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			conversation: []Message{
				{Role: "user", Content: Content{Raw: "What is the capital of France?"}},
				{Role: "assistant", Content: Content{Raw: "The capital of France is Paris."}},
			},
			description: "Basic question and answer conversation",
		},
		{
			name:      "Multi-turn Conversation",
			modelName: "microsoft/DialoGPT-medium",
			conversation: []Message{
				{Role: "user", Content: Content{Raw: "Hello, how are you?"}},
				{Role: "assistant", Content: Content{Raw: "I'm doing well, thank you! How can I help you today?"}},
				{Role: "user", Content: Content{Raw: "Can you tell me about machine learning?"}},
				{Role: "assistant", Content: Content{Raw: "Machine learning is a subset of artificial intelligence " +
					"that enables computers to learn and make decisions from data without being explicitly programmed."}},
			},
			description: "Multi-turn conversation with follow-up questions",
		},
		{
			name:      "System Message Conversation",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			conversation: []Message{
				{Role: "system", Content: Content{Raw: "You are a helpful AI assistant specialized in coding."}},
				{Role: "user", Content: Content{Raw: "Write a Python function to calculate fibonacci numbers."}},
				{
					Role: "assistant", Content: Content{Raw: "Here's a Python function to calculate fibonacci numbers:\n" +
						"def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
				},
			},
			description: "Conversation with system message and code generation",
		},
		{
			name:      "Simple Conversation (Repeated)",
			modelName: "ibm-granite/granite-3.3-8b-instruct",
			conversation: []Message{
				{Role: "user", Content: Content{Raw: "What is the capital of France?"}},
				{Role: "assistant", Content: Content{Raw: "The capital of France is Paris."}},
			},
			description: "Basic question and answer conversation (repeated to test render caching)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Logf("Testing: %s - %s", tt.name, tt.description)

			// Step 1: Get the model's chat template
			start := time.Now()
			templateRequest := preprocessing.FetchChatTemplateRequest{
				Model: tt.modelName,
			}
			template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), templateRequest)
			templateDuration := time.Since(start)
			require.NoError(t, err, "Failed to get model chat template")
			assert.NotEmpty(t, template, "ChatTemplate should not be empty")

			// Step 2: Render the conversation using the template
			start = time.Now()
			renderRequest := &RenderJinjaTemplateRequest{
				Messages:           tt.conversation,
				ChatTemplate:       template,
				ChatTemplateKWArgs: templateVars,
			}
			response, err := wrapper.RenderChatTemplate(context.Background(), renderRequest)
			renderDuration := time.Since(start)
			require.NoError(t, err, "Failed to render chat template")
			assert.NotNil(t, response, "Response should not be nil")
			assert.NotEmpty(t, response.RenderedChats, "Rendered chats should not be empty")

			// Step 3: Verify the rendered output
			rendered := response.RenderedChats[0]
			assert.NotEmpty(t, rendered, "Rendered chat should not be empty")

			// Verify all conversation messages are present in the rendered output
			for _, message := range tt.conversation {
				assert.Contains(t, rendered, message.Content.Raw, "Rendered content should contain message content")
			}

			// Log performance metrics
			t.Logf("ChatTemplate fetch duration: %v, Render duration: %v, Total duration: %v",
				templateDuration, renderDuration, templateDuration+renderDuration)
		})
	}
}

// TestVLLMValidation tests that our chat template rendering matches vLLM's expected output.
func TestVLLMValidation(t *testing.T) {
	// Test IBM Granite model
	t.Run("IBM_Granite", func(t *testing.T) {
		expectedVLLMOutput := "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\n" +
			"Today's Date: August 06, 2025.\n" +
			"You are Granite, developed by IBM. Write the response to the user's input by strictly aligning with the " +
			"facts in the provided documents. " +
			"If the information needed to answer the question is not available in the documents, inform the user that " +
			"the question cannot be answered based on the available data.<|end_of_text|>\n" +
			"<|start_of_role|>document {\"document_id\": \"\"}<|end_of_role|>\n" +
			"The weather in Paris is sunny and 25°C.<|end_of_text|>\n" +
			"<|start_of_role|>user<|end_of_role|>What is the weather in Paris?<|end_of_text|>\n" +
			"<|start_of_role|>assistant<|end_of_role|>Let me check that for you.<|end_of_text|>\n" +
			"<|start_of_role|>assistant<|end_of_role|>"
		runVLLMValidationTest(t, "ibm-granite/granite-3.3-8b-instruct", expectedVLLMOutput)
	})

	// Test TinyLlama model
	t.Run("TinyLlama", func(t *testing.T) {
		expectedVLLMOutput := "<|user|>\nWhat is the weather in Paris?</s>\n<|assistant|>\nLet me check that for you." +
			"</s>\n<|assistant|>\n"
		runVLLMValidationTest(t, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", expectedVLLMOutput)
	})
}

// TestLongChatCompletions tests with longer, more complex conversations.
func TestLongChatCompletions(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	// Create a long conversation
	longConversation := []Message{
		{
			Role: "system",
			Content: Content{Raw: "You are an expert software engineer with deep knowledge of Go, Python, " +
				"and system design. Provide detailed, accurate responses."},
		},
		{
			Role: "user",
			Content: Content{
				Raw: "I'm building a high-performance caching system in Go. Can you help me design " +
					"the architecture?",
			},
		},
		{
			Role: "assistant",
			Content: Content{Raw: "Absolutely! For a high-performance caching system in Go, I'd recommend " +
				"starting with a layered architecture. Let's break this down into components."},
		},
		{Role: "user", Content: Content{Raw: "What about memory management and eviction policies?"}},
		{
			Role: "assistant",
			Content: Content{Raw: "Great question! Memory management is crucial. I'd suggest implementing an " +
				"LRU (Least Recently Used) eviction policy " +
				"with configurable memory limits. You can use a combination of a hash map for O(1) lookups and a " +
				"doubly-linked list for tracking access order."},
		},
		{Role: "user", Content: Content{Raw: "How should I handle concurrent access and thread safety?"}},
		{
			Role: "assistant",
			Content: Content{Raw: "For thread safety, you have several options. The most common approach is " +
				"to use sync.RWMutex for read-write locks, " +
				"allowing multiple concurrent readers but exclusive writers. Alternatively, you could use sync.Map " +
				"for simpler cases or implement a lock-free design " +
				"with atomic operations for maximum performance."},
		},
		{Role: "user", Content: Content{Raw: "What about persistence and recovery?"}},
		{
			Role: "assistant",
			Content: Content{Raw: "For persistence, consider using a write-ahead log (WAL) pattern. This " +
				"involves logging all mutations to disk before applying them to memory. " +
				"For recovery, you can replay the log to reconstruct the cache state. You might also want to " +
				"implement periodic snapshots for faster recovery."},
		},
	}

	modelName := "ibm-granite/granite-3.3-8b-instruct"

	t.Run("Long Conversation Processing", func(t *testing.T) {
		// Get template
		start := time.Now()
		templateRequest := preprocessing.FetchChatTemplateRequest{
			Model: modelName,
		}
		template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), templateRequest)
		templateDuration := time.Since(start)
		require.NoError(t, err, "Failed to get model chat template")

		// Render long conversation
		start = time.Now()
		renderRequest := &RenderJinjaTemplateRequest{
			Messages:           longConversation,
			ChatTemplate:       template,
			ChatTemplateKWArgs: templateVars,
		}
		response, err := wrapper.RenderChatTemplate(context.Background(), renderRequest)
		renderDuration := time.Since(start)
		require.NoError(t, err, "Failed to render long conversation")

		// Verify results
		rendered := response.RenderedChats[0]
		assert.NotEmpty(t, rendered, "Long conversation should render successfully")
		assert.Greater(t, len(rendered), 1000,
			"Long conversation should produce substantial output")

		// Performance metrics
		t.Logf("ChatTemplate fetch: %v, Long conversation render: %v, Total processing time: %v",
			templateDuration, renderDuration, templateDuration+renderDuration)

		// Verify all messages are present
		for _, message := range longConversation {
			assert.Contains(t, rendered, message.Content.Raw,
				"All message content should be present in rendered output")
		}
	})
}

// BenchmarkGetModelChatTemplate benchmarks the template fetching performance.
func BenchmarkGetModelChatTemplate(b *testing.B) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(b, err, "Failed to clear caches")

	request := preprocessing.FetchChatTemplateRequest{
		Model: "ibm-granite/granite-3.3-8b-instruct",
	}

	// Track first iteration time and total time
	var firstIterationTime time.Duration
	var totalTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		_, _, err := wrapper.FetchChatTemplate(context.Background(), request)
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

// BenchmarkRenderJinjaTemplate benchmarks the template rendering performance.
func BenchmarkRenderJinjaTemplate(b *testing.B) {
	wrapper := getGlobalWrapper()

	// Clear caches to ensure accurate timing measurements
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(b, err, "Failed to clear caches")

	// Get template first
	templateRequest := preprocessing.FetchChatTemplateRequest{
		Model: "ibm-granite/granite-3.3-8b-instruct",
	}
	template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), templateRequest)
	require.NoError(b, err, "Failed to get template for benchmark")

	request := &RenderJinjaTemplateRequest{
		Messages: []Message{
			{Role: "user", Content: Content{Raw: "Hello"}},
			{Role: "assistant", Content: Content{Raw: "Hi there!"}},
		},
		ChatTemplate:       template,
		ChatTemplateKWArgs: templateVars,
	}

	// Track first iteration time and total time
	var firstIterationTime time.Duration
	var totalTime time.Duration

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		start := time.Now()
		_, err := wrapper.RenderChatTemplate(context.Background(), request)
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

// Helper function.
func minLength(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// normalizeDateInOutput replaces the date in the output with the expected date for comparison.
// This is for TestVLLMValidation2, which tests Granite, that has a system prompt with today's date.
func normalizeDateInOutput(output, expected string) string {
	// Find the date pattern in our output: "Today's Date: " followed by date and ".\n"
	datePattern := "Today's Date: "
	startIdx := strings.Index(output, datePattern)
	if startIdx == -1 {
		return output // No date found, return as is
	}

	// Find the end of the date (before ".\n")
	endPattern := ".\n"
	endIdx := strings.Index(output[startIdx:], endPattern)
	if endIdx == -1 {
		return output // No end pattern found, return as is
	}

	// Find the expected date in the expected output
	expectedStartIdx := strings.Index(expected, datePattern)
	if expectedStartIdx == -1 {
		return output // No expected date found, return as is
	}

	expectedEndIdx := strings.Index(expected[expectedStartIdx:], endPattern)
	if expectedEndIdx == -1 {
		return output // No expected end pattern found, return as is
	}

	// Extract the expected date
	expectedDate := expected[expectedStartIdx : expectedStartIdx+expectedEndIdx+len(endPattern)]

	// Replace our date with the expected date
	beforeDate := output[:startIdx]
	afterDate := output[startIdx+endIdx+len(endPattern):]

	return beforeDate + expectedDate + afterDate
}

// runVLLMValidationTest runs a vLLM validation test with the given model and expected output.
func runVLLMValidationTest(t *testing.T, modelName, expectedVLLMOutput string) {
	t.Helper()
	wrapper := getGlobalWrapper()

	// Test case based on the provided vLLM request
	request := &RenderJinjaTemplateRequest{
		Messages: []Message{
			{Role: "user", Content: Content{Raw: "What is the weather in Paris?"}},
			{Role: "assistant", Content: Content{Raw: "Let me check that for you."}},
		},
		Documents: []interface{}{
			map[string]interface{}{
				"title": "Paris Weather Report",
				"text":  "The weather in Paris is sunny and 25°C.",
			},
		},
		ChatTemplate: "", // Will be fetched from model
		ChatTemplateKWArgs: map[string]interface{}{
			"max_tokens":  10,
			"temperature": 0.0,
		},
	}

	// Step 1: Get the chat template from the specified model
	templateRequest := preprocessing.FetchChatTemplateRequest{
		Model: modelName,
	}
	template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), templateRequest)
	require.NoError(t, err, "Failed to get chat template")
	assert.NotEmpty(t, template, "ChatTemplate should not be empty")

	// Step 2: Update the request with the actual template and template variables
	request.ChatTemplate = template
	if templateVars != nil {
		// Use the template variables from the model (contains special tokens like eos_token)
		request.ChatTemplateKWArgs = templateVars
	}

	// Step 3: Render the conversation with the template
	response, err := wrapper.RenderChatTemplate(context.Background(), request)
	require.NoError(t, err, "Failed to render chat template")
	require.Len(t, response.RenderedChats, 1, "Should have one rendered chat")

	renderedOutput := response.RenderedChats[0]

	// Step 4: Compare results with flexible date handling
	compareVLLMOutput(t, renderedOutput, expectedVLLMOutput)
}

// compareVLLMOutput compares our rendered output with expected vLLM output and reports the result.
func compareVLLMOutput(t *testing.T, renderedOutput, expectedVLLMOutput string) {
	t.Helper()
	// Create a flexible comparison that handles dynamic dates
	// Replace the date in our output with the expected date for comparison
	normalizedOutput := normalizeDateInOutput(renderedOutput, expectedVLLMOutput)

	// Option 1: Perfect duplicates (after date normalization)
	if normalizedOutput == expectedVLLMOutput {
		t.Log("✅ PERFECT MATCH: Our output exactly matches vLLM expected output (after date normalization)")
		return
	}

	// Option 2: Perfect prefix (our output is a prefix of expected, after date normalization)
	if strings.HasPrefix(expectedVLLMOutput, normalizedOutput) {
		suffix := expectedVLLMOutput[len(normalizedOutput):]
		t.Logf("✅ PERFECT PREFIX: Our output is a perfect prefix of vLLM expected output (after date normalization). "+
			"Missing suffix: %q. This might be expected if vLLM adds additional tokens", suffix)
		return
	}

	// Option 3: Neither - failed result
	t.Logf("❌ FAILED: Our output does not match vLLM expected output (even after date normalization). "+
		"Our output length: %d, Expected length: %d, Normalized output: %q",
		len(renderedOutput), len(expectedVLLMOutput), normalizedOutput)

	// Find the first difference
	minLen := minLength(len(normalizedOutput), len(expectedVLLMOutput))
	for i := 0; i < minLen; i++ {
		if normalizedOutput[i] != expectedVLLMOutput[i] {
			t.Logf("   First difference at position %d: our='%c' (0x%02x), expected='%c' (0x%02x)",
				i, normalizedOutput[i], normalizedOutput[i], expectedVLLMOutput[i], expectedVLLMOutput[i])
			break
		}
	}

	t.Fail() // Mark test as failed
}

// TestFetchChatTemplateLocalPath tests fetching chat templates from local paths.
func TestFetchChatTemplateLocalPath(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Get the path to the test model tokenizer
	// The testdata directory is in pkg/tokenization/testdata
	testModelPath := "../../tokenization/testdata/test-model"

	request := preprocessing.FetchChatTemplateRequest{
		Model:       testModelPath,
		IsLocalPath: true,
	}

	// Fetch the chat template
	template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), request)

	// Assertions
	require.NoError(t, err, "FetchChatTemplate should not return an error for local path")
	assert.NotEmpty(t, template, "ChatTemplate should not be empty")
	assert.NotNil(t, templateVars, "ChatTemplate vars should not be nil")

	// Verify the template contains expected content
	assert.Contains(t, template, "messages", "ChatTemplate should contain messages variable")
	t.Logf("Fetched local template: %s", template)
	t.Logf("Template vars: %+v", templateVars)
}

// TestRenderChatTemplateWithLocalTemplate tests rendering with a locally fetched template.
func TestRenderChatTemplateWithLocalTemplate(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Get the path to the test model tokenizer
	testModelPath := "../../tokenization/testdata/test-model"

	// First, fetch the template
	fetchRequest := preprocessing.FetchChatTemplateRequest{
		Model:       testModelPath,
		IsLocalPath: true,
	}

	template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), fetchRequest)
	require.NoError(t, err, "FetchChatTemplate should not return an error")

	// Now render a conversation using the fetched template
	renderRequest := &RenderJinjaTemplateRequest{
		Messages: []Message{
			{Role: "user", Content: Content{Raw: "Hello from local tokenizer!"}},
			{Role: "assistant", Content: Content{Raw: "Hi! I'm using a locally loaded template."}},
		},
		ChatTemplate:       template,
		ChatTemplateKWArgs: templateVars,
	}

	response, err := wrapper.RenderChatTemplate(context.Background(), renderRequest)
	require.NoError(t, err, "RenderChatTemplate should not return an error")
	assert.NotNil(t, response, "Response should not be nil")
	assert.NotEmpty(t, response.RenderedChats, "Rendered chats should not be empty")

	// Verify the rendered content
	rendered := response.RenderedChats[0]
	assert.Contains(t, rendered, "Hello from local tokenizer!", "Rendered content should contain user message")
	assert.Contains(t, rendered, "Hi! I'm using a locally loaded template.", "Rendered content should contain assistant message")

	t.Logf("Rendered chat with local template:\n%s", rendered)
}

// TestFetchChatTemplateLocalPathCaching tests that local templates are cached properly.
func TestFetchChatTemplateLocalPathCaching(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Clear caches first
	err := preprocessing.ClearCaches(context.Background())
	require.NoError(t, err, "Failed to clear caches")

	testModelPath := "../../tokenization/testdata/test-model"
	request := preprocessing.FetchChatTemplateRequest{
		Model:       testModelPath,
		IsLocalPath: true,
	}

	// First call - cache miss
	start := time.Now()
	template1, vars1, err := wrapper.FetchChatTemplate(context.Background(), request)
	duration1 := time.Since(start)
	require.NoError(t, err, "First call should not return an error")

	// Second call - cache hit
	start = time.Now()
	template2, vars2, err := wrapper.FetchChatTemplate(context.Background(), request)
	duration2 := time.Since(start)
	require.NoError(t, err, "Second call should not return an error")

	// Verify results are identical
	assert.Equal(t, template1, template2, "Cached and non-cached templates should be identical")
	assert.Equal(t, vars1, vars2, "Cached and non-cached vars should be identical")

	// Cache hit should be faster
	t.Logf("First call (cache miss): %v, Second call (cache hit): %v, Speedup: %.1fx",
		duration1, duration2, float64(duration1)/float64(duration2))
	assert.Less(t, duration2, duration1, "Cache hit should be faster than cache miss")
}

// TestFetchChatTemplateLocalPathWithFile tests loading from a specific tokenizer.json file path.
func TestFetchChatTemplateLocalPathWithFile(t *testing.T) {
	wrapper := getGlobalWrapper()

	// Test with the full path to tokenizer.json
	//nolint:gosec // This is a test file path, not a credential
	testTokenizerPath := "../../tokenization/testdata/test-model/tokenizer.json"

	request := preprocessing.FetchChatTemplateRequest{
		Model:       testTokenizerPath,
		IsLocalPath: true,
	}

	// Fetch the chat template - should extract directory and load from there
	template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), request)

	// Assertions
	require.NoError(t, err, "FetchChatTemplate should handle file path and extract directory")
	assert.NotEmpty(t, template, "ChatTemplate should not be empty")
	assert.NotNil(t, templateVars, "ChatTemplate vars should not be nil")
	assert.Contains(t, template, "messages", "ChatTemplate should contain messages variable")

	t.Logf("Fetched template from file path: %s", template)
}

// TestFetchChatTemplateLocalPathNonExistent tests error handling for non-existent local paths.
func TestFetchChatTemplateLocalPathNonExistent(t *testing.T) {
	wrapper := getGlobalWrapper()

	request := preprocessing.FetchChatTemplateRequest{
		Model:       "/non/existent/path",
		IsLocalPath: true,
	}

	// This should return an error
	template, templateVars, err := wrapper.FetchChatTemplate(context.Background(), request)

	// Assertions
	assert.Error(t, err, "FetchChatTemplate should return an error for non-existent path")
	assert.Empty(t, template, "ChatTemplate should be empty on error")
	assert.Nil(t, templateVars, "ChatTemplate vars should be nil on error")

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
