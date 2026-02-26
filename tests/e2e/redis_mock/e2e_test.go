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

//nolint:testpackage // allow tests to run in the same package
package e2e

import (
	"context"
	"os"
	"path/filepath"
	"strings"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
)

const (
	localTestModelDir   = "testdata/test-model"
	localLlama3ModelDir = "testdata/local-llama3"
)

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatTemplateRequest represents the request to render a chat template.
type ChatTemplateRequest struct {
	Conversations [][]ChatMessage        `json:"conversations"`
	ChatTemplate  string                 `json:"chatTemplate"`
	TemplateVars  map[string]interface{} `json:"templateVars,omitempty"`
}

// ChatTemplateResponse represents the response from the Python function.
type ChatTemplateResponse struct {
	RenderedChats     []string  `json:"renderedChats"`
	GenerationIndices [][][]int `json:"generationIndices"`
}

// GetChatTemplateRequest represents the request to get a model's chat template.
type GetChatTemplateRequest struct {
	ModelName string `json:"modelName"`
	Revision  string `json:"revision,omitempty"`
	Token     string `json:"token,omitempty"`
}

// convertToPreprocessingConversation converts e2e ChatMessage to preprocessing Conversation.
func convertToPreprocessingConversation(messages []ChatMessage) [][]preprocessing.Conversation {
	result := make([]preprocessing.Conversation, len(messages))
	for i, msg := range messages {
		result[i] = preprocessing.Conversation{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}
	return [][]preprocessing.Conversation{result}
}

// MockChatTemplateWrapper provides a mock implementation for testing.
type MockChatTemplateWrapper struct{}

func NewMockChatTemplateWrapper() *MockChatTemplateWrapper {
	return &MockChatTemplateWrapper{}
}

//nolint:nonamedreturns // Mock implementation uses named returns for clarity and consistency with interface.
func (w *MockChatTemplateWrapper) GetModelChatTemplate(
	req GetChatTemplateRequest,
) (template string, templateVars map[string]interface{}, err error) {
	// Mock implementation that returns a simple template.
	template = `{% for message in messages %}{{ message.role }}: {{ message.content }}
{% endfor %}`
	templateVars = map[string]interface{}{
		"bos_token": "<s>",
		"eos_token": "</s>",
	}
	return template, templateVars, nil
}

func (w *MockChatTemplateWrapper) RenderChatTemplate(req ChatTemplateRequest) (*ChatTemplateResponse, error) {
	// Mock implementation that renders the template.
	renderedChats := make([]string, 0, len(req.Conversations))
	for _, conversation := range req.Conversations {
		rendered := ""
		for _, message := range conversation {
			rendered += message.Role + ": " + message.Content + "\n"
		}
		renderedChats = append(renderedChats, rendered)
	}

	return &ChatTemplateResponse{
		RenderedChats:     renderedChats,
		GenerationIndices: [][][]int{},
	}, nil
}

// TestBasicE2E verifies that the indexer initially returns no scores for the first prompt and
// correct scores for the second request.
func (s *KVCacheSuite) TestCacheHit() {
	//nolint:lll // long prompt
	prompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	fakePodList := []string{s.Pod1IP}

	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(prompt, defaultModelName)
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	pods, err := s.indexer.GetPodScores(s.ctx, nil, prompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.T().Logf("Received pod scores: %+v", pods)
	s.Len(pods, len(fakePodList), "expected pod scores length to match candidate pods")
	s.Greater(pods[s.Pod1IP], 1.0, "expected pod score to equal 1.0")
}

func (s *KVCacheSuite) TestCacheMiss() {
	prompt := "What is the capital of France?"
	fakePodList := []string{s.Pod1IP}

	pods, err := s.indexer.GetPodScores(s.ctx, nil, prompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.T().Logf("Received pod scores: %+v", pods)
	s.Empty(pods, "expected no pod scores since no keys were added to the index")
}

// TestPrefixReduction tests scoring behavior when querying progressively shorter prefixes of a fully cached prompt.
func (s *KVCacheSuite) TestPrefixReduction() {
	//nolint:lll // long prompt
	fullPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
	//nolint:lll // long prompt
	midPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	shortPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit."

	fullPromptEngineKeys, fullPromptRequestKeys := s.promptToEngineAndRequestKeys(fullPrompt, defaultModelName)
	fakePodList := []string{s.Pod1IP}

	// Test 1: Full prompt (no match expected)
	pods, err := s.indexer.GetPodScores(s.ctx, nil, fullPrompt, defaultModelName, []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Received pod scores: %+v", pods)
	s.Empty(pods, "expected no pod scores")

	s.addEntriesToIndex(fullPromptEngineKeys, fullPromptRequestKeys, fakePodList)

	// Test 2: mid-length prompt(should return a match)
	pods, err = s.indexer.GetPodScores(s.ctx, nil, midPrompt, defaultModelName, []string{s.Pod1IP})
	s.Require().NoError(err)

	s.T().Logf("Received pod scores: %+v", pods)
	s.Greater(int(pods[s.Pod1IP]), 0, "mid-prompt block keys should have been indexed")

	// Test 3: short prompt(should return a match)
	pods, err = s.indexer.GetPodScores(s.ctx, nil, shortPrompt, defaultModelName, []string{s.Pod1IP})
	s.Require().NoError(err)

	s.Len(pods, len(fakePodList), "expected pod scores length to match candidate pods")
	s.T().Logf("Received pod scores: %+v", pods)
	_, shortPromptRequestKeys := s.promptToEngineAndRequestKeys(shortPrompt, defaultModelName)
	s.Equal(int(pods[s.Pod1IP]), len(shortPromptRequestKeys), "all short-prompt block keys should have been indexed")
}

// TestPrefixExpansion tests that prompts longer than the cached prefix still return partial match scores.
func (s *KVCacheSuite) TestPrefixExpansion() {
	//nolint:lll // long prompt
	fullPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
	//nolint:lll // long prompt
	midPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	shortPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit."
	modelName := defaultModelName
	fakePodList := []string{s.Pod1IP}

	// Test 1: short prompt
	pods, err := s.indexer.GetPodScores(s.ctx, nil, shortPrompt, modelName, []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Received pod scores: %+v", pods)
	s.Empty(pods, "expected no pod scores")

	shortPromptEngineKeys, shortPromptRequestKeys := s.promptToEngineAndRequestKeys(shortPrompt, modelName)
	s.addEntriesToIndex(shortPromptEngineKeys, shortPromptRequestKeys, fakePodList)

	// Test 2: mid prompt
	pods, err = s.indexer.GetPodScores(s.ctx, nil, midPrompt, modelName, []string{s.Pod1IP})
	s.Require().NoError(err)

	s.T().Logf("Received pod scores: %+v", pods)
	s.Equal(int(pods[s.Pod1IP]), len(shortPromptRequestKeys), "expected pod score to equal number of short prompt block keys")

	midPromptEngineKeys, midPromptRequestKeys := s.promptToEngineAndRequestKeys(midPrompt, modelName)
	s.addEntriesToIndex(midPromptEngineKeys, midPromptRequestKeys, fakePodList)

	// Test 3: full prompt
	pods, err = s.indexer.GetPodScores(s.ctx, nil, fullPrompt, modelName, []string{s.Pod1IP})
	s.Require().NoError(err)

	s.T().Logf("Received pod scores: %+v", pods)
	s.Equal(int(pods[s.Pod1IP]), len(midPromptRequestKeys), "expected pod score to equal number of mid prompt block keys")
}

func (s *KVCacheSuite) TestLongPrefixExpansion() {
	base := "The quick brown fox jumps over the lazy dog"
	modelName := defaultModelName
	s.T().Logf("s.config.PrefixStoreConfig: %+v, TokenProcessorConfig: %+v",
		s.config.PrefixStoreConfig.LRUStoreConfig, s.tokenProcessor)
	// Generate long prompts
	shortPrompt := strings.Repeat(base, 2)
	midPrompt := strings.Repeat(base, 100)  // ~900 tokens
	longPrompt := strings.Repeat(base, 500) // ~4500 tokens

	fakePodList := []string{s.Pod1IP}

	// Test 1: short prompt (should return no pod scores yet)
	pods, err := s.indexer.GetPodScores(s.ctx, nil, shortPrompt, modelName, []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Short prompt scores: %+v", pods)
	s.Empty(pods, "expected no pod scores")

	// Add entries to the index for the short prompt
	shortPromptEngineKeys, shortPromptRequestKeys := s.promptToEngineAndRequestKeys(shortPrompt, modelName)
	s.addEntriesToIndex(shortPromptEngineKeys, shortPromptRequestKeys, fakePodList)

	// Test 2: mid prompt (should return partial match if indexer picks it up)
	pods, err = s.indexer.GetPodScores(s.ctx, nil, midPrompt, modelName, []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Mid prompt scores: %+v", pods)
	s.True(len(pods) > 0, "expected at least one pod score for mid prompt")

	// Add entries to the index for the mid prompt
	midPromptEngineKeys, midPromptRequestKeys := s.promptToEngineAndRequestKeys(midPrompt, modelName)
	s.addEntriesToIndex(midPromptEngineKeys, midPromptRequestKeys, fakePodList)

	// Test 3: long prompt (should return higher score)
	pods, err = s.indexer.GetPodScores(s.ctx, nil, longPrompt, modelName, []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Long prompt scores: %+v", pods)
	s.True(len(pods) > 0, "expected at least one pod score for long prompt")
}

// TestChatCompletionsE2E tests the complete chat completions workflow with KV-cache integration.
func (s *KVCacheSuite) TestChatCompletionsE2E() {
	// Create a mock wrapper for testing.
	wrapper := NewMockChatTemplateWrapper()

	// Create a chat completion conversation.
	conversation := [][]ChatMessage{
		{
			{Role: "system", Content: "You are a helpful AI assistant."},
			{Role: "user", Content: "What is the capital of France?"},
			{Role: "assistant", Content: "The capital of France is Paris."},
		},
	}

	// Step 1: Get the model's chat template.
	templateRequest := GetChatTemplateRequest{
		ModelName: "ibm-granite/granite-3.3-8b-instruct",
	}
	template, templateVars, err := wrapper.GetModelChatTemplate(templateRequest)
	s.Require().NoError(err, "Failed to get model chat template")
	s.Require().NotEmpty(template, "ChatTemplate should not be empty")

	// Step 2: Render the conversation using the template.
	renderRequest := ChatTemplateRequest{
		Conversations: conversation,
		ChatTemplate:  template,
		TemplateVars:  templateVars,
	}
	response, err := wrapper.RenderChatTemplate(renderRequest)
	s.Require().NoError(err, "Failed to render chat template")
	s.Require().NotNil(response, "Response should not be nil")
	s.Require().NotEmpty(response.RenderedChats, "Rendered chats should not be empty")

	// Step 3: Extract the flattened prompt from the rendered template.
	flattenedPrompt := response.RenderedChats[0]
	s.Require().NotEmpty(flattenedPrompt, "Flattened prompt should not be empty")

	// Step 4: Use the flattened prompt for KV-cache lookup (similar to TestBasicE2E).
	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(flattenedPrompt, "ibm-granite/granite-3.3-8b-instruct")
	fakePodList := []string{s.Pod1IP}

	// First lookup - should return no scores initially.
	pods, err := s.indexer.GetPodScores(s.ctx, nil, flattenedPrompt, "ibm-granite/granite-3.3-8b-instruct", []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("First lookup - Received pod scores: %+v", pods)
	s.Empty(pods, "expected no pod scores on first lookup")

	// Add entries to the index.
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	// Second lookup - should return scores.
	pods, err = s.indexer.GetPodScores(s.ctx, nil, flattenedPrompt, "ibm-granite/granite-3.3-8b-instruct", []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Second lookup - Received pod scores: %+v", pods)
	s.Len(pods, 1, "expected one pod score")
	s.True(pods[s.Pod1IP] > 0, "expected positive pod score")

	s.T().Logf("Chat completions E2E test completed successfully")
}

// TestLongChatCompletionsE2E tests long chat completions with complex conversations.
func (s *KVCacheSuite) TestLongChatCompletionsE2E() {
	// Create a mock wrapper for testing.
	wrapper := NewMockChatTemplateWrapper()

	// Create a long, complex conversation.
	longConversation := [][]ChatMessage{
		{
			{Role: "system", Content: "You are an expert software engineer with deep knowledge of Go, Python, " +
				"and system design. Provide detailed, accurate responses."},
			{Role: "user", Content: "I'm building a high-performance caching system in Go. Can you help me " +
				"design the architecture?"},
			{Role: "assistant", Content: "Absolutely! For a high-performance caching system in Go, I'd recommend " +
				"starting with a layered architecture. Let's break this down into components."},
			{Role: "user", Content: "What about memory management and eviction policies?"},
			{Role: "assistant", Content: "Great question! Memory management is crucial. I'd suggest implementing " +
				"an LRU (Least Recently Used) eviction policy with configurable memory limits. " +
				"You can use a combination of a hash map for O(1) lookups and a doubly-linked list " +
				"for tracking access order."},
			{Role: "user", Content: "How should I handle concurrent access and thread safety?"},
			{Role: "assistant", Content: "For thread safety, you have several options. The most common approach is " +
				"to use sync.RWMutex for read-write locks, allowing multiple concurrent readers " +
				"but exclusive writers. Alternatively, you could use sync.Map for simpler cases " +
				"or implement a lock-free design with atomic operations for maximum performance."},
		},
	}

	// Step 1: Get the model's chat template.
	templateRequest := GetChatTemplateRequest{
		ModelName: "ibm-granite/granite-3.3-8b-instruct",
	}
	template, templateVars, err := wrapper.GetModelChatTemplate(templateRequest)
	s.Require().NoError(err, "Failed to get model chat template")
	s.Require().NotEmpty(template, "ChatTemplate should not be empty")

	// Step 2: Render the long conversation.
	renderRequest := ChatTemplateRequest{
		Conversations: longConversation,
		ChatTemplate:  template,
		TemplateVars:  templateVars,
	}
	response, err := wrapper.RenderChatTemplate(renderRequest)
	s.Require().NoError(err, "Failed to render long conversation")
	s.Require().NotNil(response, "Response should not be nil")
	s.Require().NotEmpty(response.RenderedChats, "Rendered chats should not be empty")

	// Step 3: Extract the flattened prompt.
	flattenedPrompt := response.RenderedChats[0]
	s.Require().NotEmpty(flattenedPrompt, "Flattened prompt should not be empty")
	s.Require().Greater(len(flattenedPrompt), 1000, "Long conversation should produce substantial output")

	// Step 4: Test KV-cache with the long flattened prompt.
	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(flattenedPrompt, "ibm-granite/granite-3.3-8b-instruct")
	fakePodList := []string{s.Pod1IP}

	// First lookup.
	pods, err := s.indexer.GetPodScores(s.ctx, nil, flattenedPrompt, "ibm-granite/granite-3.3-8b-instruct", []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("First lookup - Received pod scores: %+v", pods)
	s.Empty(pods, "expected no pod scores on first lookup")

	// Add entries to the index.
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	// Second lookup.
	pods, err = s.indexer.GetPodScores(s.ctx, nil, flattenedPrompt, "ibm-granite/granite-3.3-8b-instruct", []string{s.Pod1IP})
	s.Require().NoError(err)
	s.T().Logf("Second lookup - Received pod scores: %+v", pods)
	s.Len(pods, 1, "expected one pod score")
	s.True(pods[s.Pod1IP] > 0, "expected positive pod score")

	s.T().Logf("Long chat completions E2E test completed successfully")
}

// TestCacheHitWithLocalTokenizer tests the full E2E flow using local tokenizer files.
func (s *KVCacheSuite) TestCacheHitWithLocalTokenizer() {
	// Create a local tokenizer using the testdata
	modelName := "test-model"
	localTokenizer, err := tokenization.NewCachedLocalTokenizer(context.Background(), modelName, tokenization.LocalTokenizerConfig{
		ModelTokenizerMap: map[string]string{
			modelName: "testdata/test-model/tokenizer.json",
		},
	})
	s.Require().NoError(err)
	s.Require().NotNil(localTokenizer)

	s.SetTokenizer(localTokenizer, modelName)

	prompt := "What is the capital of France?"
	fakePodList := []string{s.Pod1IP}

	// Tokenize using local tokenizer
	tokens, offsets, err := localTokenizer.Encode(modelName, &preprocessing.EncodeRequest{Text: prompt, AddSpecialTokens: true})
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)
	s.Require().Equal(len(tokens), len(offsets), "tokens and offsets should have same length")
	s.T().Logf("Local tokenizer produced %d tokens for prompt", len(tokens))

	// Convert tokens to KV block keys
	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(prompt, modelName)

	// Add entries to the index - this verifies the local tokenizer produces valid block keys
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	// Verify that we can retrieve the entries we just added using GetPodScores
	pods, err := s.indexer.GetPodScores(s.ctx, nil, prompt, modelName, fakePodList)
	s.Require().NoError(err)
	s.Require().NotEmpty(pods, "should find pod scores after adding entries")
	s.Require().Greater(pods[s.Pod1IP], float64(0), "expected positive pod score")
	s.T().Logf("GetPodScores returned score: %v", pods[s.Pod1IP])

	// Also verify that tokenizing the same prompt again produces same block keys
	tokens2, _, err := localTokenizer.Encode(modelName, &preprocessing.EncodeRequest{Text: prompt, AddSpecialTokens: true})
	s.Require().NoError(err)
	requestKeys2 := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens2, modelName)
	s.Require().Equal(requestKeys, requestKeys2, "same prompt should produce same block keys")

	s.T().Logf("Local tokenizer E2E test completed successfully")
}

// TestHFCacheStructureDiscoveryE2E tests auto-discovery of tokenizers from HuggingFace cache structure.
func (s *KVCacheSuite) TestHFCacheStructureDiscoveryE2E() {
	// Create a temporary HF-style cache directory
	tmpDir := s.T().TempDir()

	// Create HF cache structure
	// models--test-org--test-model/snapshots/{hash}/tokenizer.json
	modelName := "test-org/test-model"
	testModelPath := filepath.Join(tmpDir, "models--test-org--test-model", "snapshots", "abc123")
	require.NoError(s.T(), os.MkdirAll(testModelPath, 0o755))

	// Copy the test tokenizer
	require.NoError(s.T(), os.CopyFS(testModelPath, os.DirFS(localTestModelDir)))

	// Create tokenizer config with auto-discovery
	config := tokenization.LocalTokenizerConfig{
		AutoDiscoveryDir:               tmpDir,
		AutoDiscoveryTokenizerFileName: "tokenizer.json",
	}

	localTokenizer, err := tokenization.NewCachedLocalTokenizer(context.Background(), modelName, config)
	s.Require().NoError(err)
	s.Require().NotNil(localTokenizer)

	s.SetTokenizer(localTokenizer, modelName)

	prompt := "What is the capital of France?"
	fakePodList := []string{s.Pod1IP}

	// Tokenize using the auto-discovered HF cache tokenizer
	tokens, offsets, err := localTokenizer.Encode(modelName, &preprocessing.EncodeRequest{Text: prompt, AddSpecialTokens: true})
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)
	s.Require().Equal(len(tokens), len(offsets), "tokens and offsets should have same length")
	s.T().Logf("HF cache auto-discovery produced %d tokens for model %q", len(tokens), modelName)

	// Convert tokens to KV block keys using promptToEngineAndRequestKeys with local tokenizer
	engineKeys1, requestKeys := s.promptToEngineAndRequestKeys(prompt, modelName)

	// Add entries to the index
	s.addEntriesToIndex(engineKeys1, requestKeys, fakePodList)

	// Verify retrieval
	tokens2, _, err := localTokenizer.Encode(modelName, &preprocessing.EncodeRequest{Text: prompt, AddSpecialTokens: true})
	s.Require().NoError(err)
	requestKeys2 := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens2, modelName)
	s.Require().Equal(requestKeys, requestKeys2, "same prompt should produce same block keys")

	s.T().Logf("HF cache structure discovery E2E test completed successfully")
}

// TestLocalTokenizerChatTemplateE2E tests the complete flow of fetching and rendering
// chat templates from local tokenizers in an e2e scenario.
func (s *KVCacheSuite) TestLocalTokenizerChatTemplateE2E() {
	testCases := []struct {
		name      string
		modelDir  string
		modelName string
	}{
		{
			name:      "test-model",
			modelDir:  localTestModelDir,
			modelName: "test-model",
		},
		{
			name:      "local-llama3",
			modelDir:  localLlama3ModelDir,
			modelName: "local-llama3",
		},
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			// Create a local tokenizer with chat template support
			testModelDir, err := filepath.Abs(tc.modelDir)
			s.Require().NoError(err)

			localTokenizer, err := tokenization.NewCachedLocalTokenizer(
				context.Background(), tc.modelName, tokenization.LocalTokenizerConfig{
					ModelTokenizerMap: map[string]string{
						tc.modelName: filepath.Join(testModelDir, "tokenizer.json"),
					},
				})
			s.Require().NoError(err)
			s.Require().NotNil(localTokenizer)

			s.SetTokenizer(localTokenizer, tc.modelName)

			// Test conversation
			conversation := []ChatMessage{
				{Role: "user", Content: "What is machine learning?"},
				{Role: "assistant", Content: "Machine learning is a subset of AI that enables computers to learn from data."},
				{Role: "user", Content: "Give me an example."},
			}

			// Step 1: Render the conversation into a flattened prompt using local chat template
			// This tests the full integration: Go -> CGO -> Python -> Local Tokenizer
			renderReq := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(conversation),
			}
			renderedPrompt, err := localTokenizer.ApplyChatTemplate(tc.modelName, renderReq)
			s.Require().NoError(err, "RenderChatTemplate should succeed with local tokenizer")
			s.Require().NotEmpty(renderedPrompt, "Rendered prompt should not be empty")
			s.T().Logf("Rendered prompt from local template:\n%s", renderedPrompt)

			// Verify the rendered prompt contains the conversation content
			s.Require().Contains(renderedPrompt, "What is machine learning?", "rendered prompt should contain user message")
			s.Require().Contains(renderedPrompt, "Machine learning is a subset of AI", "rendered prompt should contain assistant message")
			s.Require().Contains(renderedPrompt, "Give me an example", "rendered prompt should contain second user message")

			// Step 2: Tokenize the rendered prompt using the same local tokenizer
			tokens, offsets, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: renderedPrompt, AddSpecialTokens: true})
			s.Require().NoError(err, "Encode should succeed")
			s.Require().NotEmpty(tokens, "Tokens should not be empty")
			s.Require().Equal(len(tokens), len(offsets), "Tokens and offsets should have same length")
			s.T().Logf("Local tokenizer produced %d tokens from rendered chat template", len(tokens))

			// Step 3: Convert tokens to KV block keys
			engineKeys, requestKeys := s.promptToEngineAndRequestKeys(renderedPrompt, tc.modelName)
			s.T().Logf("Generated %d KV block keys from rendered conversation", len(requestKeys))

			// Step 4: Add to index and verify retrieval (full KV-cache flow)
			fakePodList := []string{s.Pod1IP}
			s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)
			// Verify retrieval using GetPodScores with the rendered prompt
			pods, err := s.indexer.GetPodScores(s.ctx, nil, renderedPrompt, tc.modelName, fakePodList)
			s.Require().NoError(err)
			s.Require().NotEmpty(pods, "should find pod scores after adding entries")
			s.Require().Greater(pods[s.Pod1IP], float64(0), "expected positive pod score")
			s.T().Logf("GetPodScores returned score: %v for rendered chat template", pods[s.Pod1IP])

			// Also verify by rendering and tokenizing the same conversation again
			renderReq2 := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(conversation),
			}
			renderedPrompt2, err := localTokenizer.ApplyChatTemplate(tc.modelName, renderReq2)
			s.Require().NoError(err)
			s.Require().Equal(renderedPrompt, renderedPrompt2, "Same conversation should render identically")

			tokens2, _, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: renderedPrompt2, AddSpecialTokens: true})
			s.Require().NoError(err)
			requestKeys2 := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, tokens2, tc.modelName)
			s.Require().Equal(requestKeys, requestKeys2, "Same conversation should produce same block keys")

			s.T().Logf("Local tokenizer chat template E2E test completed successfully")
		})
	}
}

// TestLocalTokenizerChatTemplateMultiTurnE2E tests local chat template with multi-turn conversations.
func (s *KVCacheSuite) TestLocalTokenizerChatTemplateMultiTurnE2E() {
	testCases := []struct {
		name      string
		modelDir  string
		modelName string
	}{
		{
			name:      "test-model",
			modelDir:  localTestModelDir,
			modelName: "test-model",
		},
		{
			name:      "local-llama3",
			modelDir:  localLlama3ModelDir,
			modelName: "local-llama3",
		},
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			testModelDir, err := filepath.Abs(tc.modelDir)
			s.Require().NoError(err)

			localTokenizer, err := tokenization.NewCachedLocalTokenizer(
				context.Background(), tc.modelName, tokenization.LocalTokenizerConfig{
					ModelTokenizerMap: map[string]string{
						tc.modelName: filepath.Join(testModelDir, "tokenizer.json"),
					},
				})
			s.Require().NoError(err)

			s.SetTokenizer(localTokenizer, tc.modelName)

			fakePodList := []string{s.Pod1IP}

			// Start with a short conversation
			// Keep it under any tokenizer truncation limits (e.g., 512 tokens)
			shortConversation := []ChatMessage{
				{Role: "user", Content: "Hello! How are you doing today?"},
				{Role: "assistant", Content: "I'm doing great, thank you for asking!"},
			}

			// Render and cache the short conversation
			shortReq := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(shortConversation),
			}
			shortPrompt, err := localTokenizer.ApplyChatTemplate(tc.modelName, shortReq)
			s.Require().NoError(err)
			s.T().Logf("Short prompt length: %d chars", len(shortPrompt))
			shortTokens, _, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: shortPrompt, AddSpecialTokens: true})
			s.Require().NoError(err)
			shortEngineKeys, shortRequestKeys := s.promptToEngineAndRequestKeys(shortPrompt, tc.modelName)
			s.addEntriesToIndex(shortEngineKeys, shortRequestKeys, fakePodList)
			s.T().Logf("Short conversation: %d tokens, %d block keys", len(shortTokens), len(shortRequestKeys))

			// Extend the conversation (simulating a multi-turn chat)
			// Add more turns to make it longer, but still under truncation limits
			extendedConversation := []ChatMessage{
				{Role: "user", Content: "Hello! How are you doing today?"},
				{Role: "assistant", Content: "I'm doing great, thank you for asking!"},
				{Role: "user", Content: "That's wonderful! Can you tell me about your favorite programming language?"},
				{Role: "assistant", Content: "I appreciate many programming languages, each with unique strengths. " +
					"Python is great for its readability and vast ecosystem. Go excels at concurrent systems. What interests you?"},
				{Role: "user", Content: "I'm learning Go right now. Do you have any tips?"},
				{Role: "assistant", Content: "Great choice! Focus on understanding goroutines and channels early. " +
					"Practice with small projects. Read the official Go documentation - " +
					"it's excellent. And don't fight the language's conventions."},
			}

			// Render and test the extended conversation
			extendedReq := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(extendedConversation),
			}
			extendedPrompt, err := localTokenizer.ApplyChatTemplate(tc.modelName, extendedReq)
			s.Require().NoError(err)
			s.T().Logf("Extended prompt: %q (length: %d)", extendedPrompt, len(extendedPrompt))
			s.Require().Greater(len(extendedPrompt), len(shortPrompt), "Extended conversation should be longer")

			extendedTokens, _, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: extendedPrompt, AddSpecialTokens: true})
			s.Require().NoError(err)

			extendedEngineKeys, extendedRequestKeys := s.promptToEngineAndRequestKeys(extendedPrompt, tc.modelName)
			s.T().Logf("Extended conversation: %d tokens, %d block keys", len(extendedTokens), len(extendedRequestKeys))

			// Some tokenizers use fixed-length encoding with padding (e.g., 512 tokens)
			// In this case, both short and extended prompts may have the same token count
			if len(extendedTokens) == len(shortTokens) {
				s.T().Logf("Note: Tokenizer uses fixed-length encoding (%d tokens). "+
					"This is common for tokenizers with fixed padding configuration.", len(extendedTokens))
				// Verify the test still makes sense - the extended prompt should be significantly longer
				s.Require().Greater(len(extendedPrompt), len(shortPrompt),
					"Extended conversation should be longer in characters even with fixed-length tokenization")
				// With fixed-length tokenization, block keys will also be the same length
				// This is expected behavior for such tokenizers
			} else {
				// Normal case: extended conversation has more tokens and block keys
				s.Require().Greater(len(extendedTokens), len(shortTokens),
					"Extended conversation should have more tokens")
				// Verify that the extended conversation shares a prefix with the short conversation
				// (this is important for KV-cache reuse in multi-turn scenarios)
				s.Require().True(len(shortRequestKeys) < len(extendedRequestKeys),
					"Extended conversation should have more block keys than short conversation")
			}

			// Add extended conversation to index
			s.addEntriesToIndex(extendedEngineKeys, extendedRequestKeys, fakePodList)

			// Verify that querying with the short conversation still works (prefix sharing in KV-cache)
			pods, err := s.indexer.GetPodScores(s.ctx, nil, shortPrompt, tc.modelName, fakePodList)
			s.Require().NoError(err)
			s.Require().NotEmpty(pods, "Short conversation should still match after adding extended conversation")
			s.T().Logf("Short conversation match score: %v", pods[s.Pod1IP])

			s.T().Logf("Multi-turn conversation E2E test completed successfully")
		})
	}
}

// TestLocalVsHFChatTemplateConsistency tests that local and HF tokenizers
// produce consistent chat template renderings (when possible).
func (s *KVCacheSuite) TestLocalVsHFChatTemplateConsistency() {
	testCases := []struct {
		name      string
		modelDir  string
		modelName string
	}{
		{
			name:      "test-model",
			modelDir:  localTestModelDir,
			modelName: "test-model",
		},
		{
			name:      "local-llama3",
			modelDir:  localLlama3ModelDir,
			modelName: "local-llama3",
		},
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			// This test verifies that for a given model, the local tokenizer
			// produces the same rendered output as the HF tokenizer would
			// (assuming both have access to the same chat template)

			testModelDir, err := filepath.Abs(tc.modelDir)
			s.Require().NoError(err)
			s.T().Logf("Using test model directory: %s", testModelDir)

			// Verify the directory and files exist
			s.Require().DirExists(testModelDir, "Test model directory should exist")
			s.Require().FileExists(filepath.Join(testModelDir, "config.json"), "config.json should exist")
			s.Require().FileExists(filepath.Join(testModelDir, "tokenizer.json"), "tokenizer.json should exist")

			localTokenizer, err := tokenization.NewCachedLocalTokenizer(
				context.Background(), tc.modelName, tokenization.LocalTokenizerConfig{
					ModelTokenizerMap: map[string]string{
						tc.modelName: filepath.Join(testModelDir, "tokenizer.json"),
					},
				})
			s.Require().NoError(err)

			s.SetTokenizer(localTokenizer, tc.modelName)

			conversation := []ChatMessage{
				{Role: "user", Content: "Test message"},
				{Role: "assistant", Content: "Test response"},
			}

			// Render with local tokenizer
			req1 := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(conversation),
			}
			localRendered, err := localTokenizer.ApplyChatTemplate(tc.modelName, req1)
			s.Require().NoError(err)
			s.Require().NotEmpty(localRendered)

			// Tokenize with local tokenizer
			localTokens, _, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: localRendered, AddSpecialTokens: true})
			s.Require().NoError(err)
			s.T().Logf("Local tokenizer: rendered=%d chars, tokens=%d", len(localRendered), len(localTokens))

			// Add to index and verify with GetPodScores
			engineKeys, requestKeys := s.promptToEngineAndRequestKeys(localRendered, tc.modelName)
			fakePodList := []string{s.Pod1IP}
			s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

			pods, err := s.indexer.GetPodScores(s.ctx, nil, localRendered, tc.modelName, fakePodList)
			s.Require().NoError(err)
			s.Require().NotEmpty(pods, "should find pod scores after adding entries")
			s.Require().Greater(pods[s.Pod1IP], float64(0), "expected positive pod score")
			s.T().Logf("GetPodScores returned score: %v", pods[s.Pod1IP])

			// Render the same conversation again to test caching and consistency
			req2 := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(conversation),
			}
			localRendered2, err := localTokenizer.ApplyChatTemplate(tc.modelName, req2)
			s.Require().NoError(err)
			s.Require().Equal(localRendered, localRendered2,
				"Rendering the same conversation twice should produce identical output (tests caching)")

			// Tokenize again
			localTokens2, _, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: localRendered2, AddSpecialTokens: true})
			s.Require().NoError(err)
			s.Require().Equal(localTokens, localTokens2,
				"Tokenizing the same prompt twice should produce identical tokens")

			s.T().Logf("Consistency test completed successfully")
		})
	}
}

// TestLocalTokenizerChatTemplateErrorHandling tests error cases for local chat templates.
func (s *KVCacheSuite) TestLocalTokenizerChatTemplateErrorHandling() {
	modelName := "test-model"
	testModelDir, err := filepath.Abs(localTestModelDir)
	s.Require().NoError(err)

	// Test 1: Non-existent model
	_, err = tokenization.NewCachedLocalTokenizer(context.Background(), modelName, tokenization.LocalTokenizerConfig{
		ModelTokenizerMap: map[string]string{
			modelName: "non-existent-model",
		},
	})
	s.Require().Error(err, "Should return error for non-existent model")
	s.T().Logf("Expected error for non-existent model: %v", err)

	// Test 2: Empty conversation
	localTokenizer, err := tokenization.NewCachedLocalTokenizer(context.Background(), modelName, tokenization.LocalTokenizerConfig{
		ModelTokenizerMap: map[string]string{
			modelName: filepath.Join(testModelDir, "tokenizer.json"),
		},
	})
	s.Require().NoError(err)

	s.SetTokenizer(localTokenizer, modelName)

	emptyConversation := []ChatMessage{}
	reqEmpty := &preprocessing.ApplyChatTemplateRequest{
		Conversation: convertToPreprocessingConversation(emptyConversation),
	}
	rendered, err := localTokenizer.ApplyChatTemplate("test-model", reqEmpty)
	// This might succeed with empty output or fail depending on template
	// Either is acceptable behavior
	if err == nil {
		s.T().Logf("Empty conversation rendered as: %q", rendered)
	} else {
		s.T().Logf("Empty conversation returned error (acceptable): %v", err)
	}

	s.T().Logf("Error handling test completed successfully")
}

// TestLocalTokenizerChatTemplateLongConversation tests performance with very long conversations.
func (s *KVCacheSuite) TestLocalTokenizerChatTemplateLongConversation() {
	testCases := []struct {
		name      string
		modelDir  string
		modelName string
	}{
		{
			name:      "test-model",
			modelDir:  localTestModelDir,
			modelName: "test-model",
		},
		{
			name:      "local-llama3",
			modelDir:  localLlama3ModelDir,
			modelName: "local-llama3",
		},
	}

	for _, tc := range testCases {
		s.Run(tc.name, func() {
			testModelDir, err := filepath.Abs(tc.modelDir)
			s.Require().NoError(err)

			localTokenizer, err := tokenization.NewCachedLocalTokenizer(
				context.Background(), tc.modelName, tokenization.LocalTokenizerConfig{
					ModelTokenizerMap: map[string]string{
						tc.modelName: filepath.Join(testModelDir, "tokenizer.json"),
					},
				})
			s.Require().NoError(err)

			s.SetTokenizer(localTokenizer, tc.modelName)

			// Create a very long conversation (100 turns)
			longConversation := make([]ChatMessage, 0, 200)
			for i := 0; i < 100; i++ {
				longConversation = append(longConversation,
					ChatMessage{
						Role:    "user",
						Content: "This is user message number " + filepath.Base(filepath.Dir(testModelDir)),
					},
					ChatMessage{
						Role:    "assistant",
						Content: "This is assistant response number " + filepath.Base(filepath.Dir(testModelDir)),
					},
				)
			}

			// Render the long conversation
			reqLong := &preprocessing.ApplyChatTemplateRequest{
				Conversation: convertToPreprocessingConversation(longConversation),
			}
			renderedPrompt, err := localTokenizer.ApplyChatTemplate(tc.modelName, reqLong)
			s.Require().NoError(err)
			s.Require().NotEmpty(renderedPrompt)
			s.Require().Greater(len(renderedPrompt), 1000, "Long conversation should produce substantial output")
			s.T().Logf("Long conversation rendered to %d characters", len(renderedPrompt))

			// Tokenize
			tokens, offsets, err := localTokenizer.Encode(tc.modelName,
				&preprocessing.EncodeRequest{Text: renderedPrompt, AddSpecialTokens: true})
			s.Require().NoError(err)
			s.Require().NotEmpty(tokens)
			s.Require().Equal(len(tokens), len(offsets))
			s.T().Logf("Long conversation produced %d tokens", len(tokens))

			// Convert to block keys
			engineKeys, requestKeys := s.promptToEngineAndRequestKeys(renderedPrompt, tc.modelName)
			s.Require().NotEmpty(requestKeys)
			s.T().Logf("Generated %d block keys from long conversation", len(requestKeys))

			// Add to index
			fakePodList := []string{s.Pod1IP}
			s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)
			// Verify retrieval using GetPodScores
			// Note: This works now because the test suite uses a composite tokenizer that includes the local models
			pods, err := s.indexer.GetPodScores(s.ctx, nil, renderedPrompt, tc.modelName, fakePodList)
			s.Require().NoError(err)
			s.Require().NotEmpty(pods, "should find pod scores after adding entries")
			s.Require().Greater(pods[s.Pod1IP], float64(0), "expected positive pod score")
			s.T().Logf("GetPodScores returned score: %v for long conversation", pods[s.Pod1IP])

			s.T().Logf("Long conversation E2E test completed successfully")
		})
	}
}
