//go:build !embedded_tokenizers

/*
Copyright 2026 The llm-d Authors.

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
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

// ---------------------------------------------------------------------------
// Tokenization-only tests
// ---------------------------------------------------------------------------

// TestTokenize verifies that the UDS tokenizer produces non-empty, deterministic results.
func (s *UDSTokenizerSuite) TestTokenize() {
	prompt := "What is the capital of France?"

	tokens1, offsets1, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens1, "token IDs should not be empty")
	s.Require().NotEmpty(offsets1, "offsets should not be empty")
	s.Require().Equal(len(tokens1), len(offsets1), "tokens and offsets should have the same length")
	s.T().Logf("Tokenized %d tokens for prompt", len(tokens1))

	// Tokenize the same prompt again — results must be identical (determinism).
	tokens2, offsets2, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().Equal(tokens1, tokens2, "repeated tokenization should be deterministic (token IDs)")
	s.Require().Equal(offsets1, offsets2, "repeated tokenization should be deterministic (offsets)")
}

// TestTokenizeWithSpecialTokens verifies that Encode(prompt, true) includes special tokens
// and Encode(prompt, false) does not.
// Uses BERT model which always adds [CLS] and [SEP] tokens for strict greater-than comparison.
func (s *UDSTokenizerSuite) TestTokenizeWithSpecialTokens() {
	// Switch to BERT model which adds [CLS] and [SEP] special tokens
	s.switchTokenizer("google-bert/bert-base-uncased")

	prompt := "Hello world"

	tokensWithSpecial, _, err := s.tokenizer.Encode(prompt, true)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokensWithSpecial)

	tokensWithoutSpecial, _, err := s.tokenizer.Encode(prompt, false)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokensWithoutSpecial)

	// BERT adds [CLS] at the start and [SEP] at the end when add_special_tokens=true.
	// So tokens with special tokens should always be strictly greater.
	s.Require().Greater(len(tokensWithSpecial), len(tokensWithoutSpecial),
		"encoding with special tokens should produce more tokens (BERT adds [CLS] and [SEP])")

	// Verify BERT-specific special token IDs
	bosTokenID := uint32(101) // [CLS]
	eosTokenID := uint32(102) // [SEP]
	s.Require().Equal(bosTokenID, tokensWithSpecial[0], "first token should be [CLS] (101)")
	s.Require().Equal(eosTokenID, tokensWithSpecial[len(tokensWithSpecial)-1], "last token should be [SEP] (102)")

	s.T().Logf("Tokens with special: %d, without special: %d", len(tokensWithSpecial), len(tokensWithoutSpecial))
}

// TestRenderChatTemplate tests rendering a multi-turn conversation via the
// model's tokenizer chat template.
func (s *UDSTokenizerSuite) TestRenderChatTemplate() {
	conversation := []types.Conversation{
		{Role: "user", Content: "What is machine learning?"},
		{Role: "assistant", Content: "Machine learning is a subset of AI."},
		{Role: "user", Content: "Give me an example."},
	}

	renderReq := &types.RenderChatRequest{
		Conversation: conversation,
	}

	tokens, offsets, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err, "RenderChat should succeed")
	s.Require().NotEmpty(tokens, "rendered tokens should not be empty")
	s.Require().Equal(len(tokens), len(offsets), "tokens and offsets length must match")
	s.T().Logf("RenderChat produced %d tokens", len(tokens))

	// Verify determinism.
	tokens2, _, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Require().Equal(tokens, tokens2, "RenderChat should be deterministic")
}

// TestInitializeBadModel tries to create a tokenizer for a non-existent model
// and verifies that it returns an error.
func (s *UDSTokenizerSuite) TestInitializeBadModel() {
	_, err := tokenization.NewUdsTokenizer(
		s.T().Context(),
		&tokenization.UdsTokenizerConfig{
			SocketFile: s.grpcAddress,
			UseTCP:     true,
		},
		"non-existent-org/non-existent-model",
	)
	s.Require().Error(err, "initializing with a non-existent model should fail")
	s.T().Logf("Expected error for bad model: %v", err)
}

// ---------------------------------------------------------------------------
// Full KV-cache flow tests
// ---------------------------------------------------------------------------

// TestCacheHit tokenizes a prompt via UDS, adds block keys to the index,
// and verifies that GetPodScores returns positive scores.
func (s *UDSTokenizerSuite) TestCacheHit() {
	//nolint:lll // long prompt
	prompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	fakePodList := []string{s.Pod1IP}

	tokens, _, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)

	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(tokens)
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	pods, err := s.indexer.GetPodScores(s.T().Context(), nil, prompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.T().Logf("Received pod scores: %+v", pods)
	s.Len(pods, len(fakePodList), "expected pod scores length to match candidate pods")
	s.Greater(pods[s.Pod1IP], 1.0, "expected positive pod score")
}

// TestCacheMiss queries scores for a prompt that has no index entries
// and verifies empty/zero scores.
func (s *UDSTokenizerSuite) TestCacheMiss() {
	prompt := "What is the capital of France?"
	fakePodList := []string{s.Pod1IP}

	pods, err := s.indexer.GetPodScores(s.T().Context(), nil, prompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.T().Logf("Received pod scores: %+v", pods)
	s.Empty(pods, "expected no pod scores since no keys were added to the index")
}

// TestPrefixReduction caches a full prompt, then queries progressively shorter
// prefixes and verifies partial-match scoring.
func (s *UDSTokenizerSuite) TestPrefixReduction() {
	//nolint:lll // long prompt
	fullPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
	//nolint:lll // long prompt
	midPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	shortPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit."

	tokens, _, err := s.tokenizer.Render(fullPrompt)
	s.Require().NoError(err)

	fullEngineKeys, fullRequestKeys := s.promptToEngineAndRequestKeys(tokens)
	fakePodList := []string{s.Pod1IP}

	// Before indexing — no match expected.
	pods, err := s.indexer.GetPodScores(s.T().Context(), nil, fullPrompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.Empty(pods, "expected no pod scores before indexing")

	s.addEntriesToIndex(fullEngineKeys, fullRequestKeys, fakePodList)

	// Mid-length prompt should match.
	pods, err = s.indexer.GetPodScores(s.T().Context(), nil, midPrompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.Greater(int(pods[s.Pod1IP]), 0, "mid-prompt should have scored > 0")
	s.T().Logf("Mid prompt scores: %+v", pods)

	// Short prompt should match.
	pods, err = s.indexer.GetPodScores(s.T().Context(), nil, shortPrompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.Len(pods, len(fakePodList), "expected pod scores for short prompt")
	s.T().Logf("Short prompt scores: %+v", pods)

	// Verify the short prompt score equals the number of its block keys.
	shortTokens, _, err := s.tokenizer.Render(shortPrompt)
	s.Require().NoError(err)
	_, shortRequestKeys := s.promptToEngineAndRequestKeys(shortTokens)
	s.Equal(int(pods[s.Pod1IP]), len(shortRequestKeys),
		"all short-prompt block keys should have been indexed")
}

// TestChatCompletionsFlow renders a chat conversation via UDS RenderChat(),
// computes block keys, adds them to the index, and queries scores — verifying
// the full chat-completions-to-scoring pipeline over UDS.
func (s *UDSTokenizerSuite) TestChatCompletionsFlow() {
	conversation := []types.Conversation{
		{Role: "system", Content: "You are a helpful AI assistant."},
		{Role: "user", Content: "What is the capital of France?"},
		{Role: "assistant", Content: "The capital of France is Paris."},
	}

	renderReq := &types.RenderChatRequest{
		Conversation: conversation,
	}

	tokens, _, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)
	s.T().Logf("Chat completions rendered %d tokens", len(tokens))

	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(tokens)
	fakePodList := []string{s.Pod1IP}

	// First lookup — no match.
	pods, err := s.indexer.GetPodScores(s.T().Context(), renderReq, "", defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.Empty(pods, "expected no pod scores on first lookup")

	// Index and lookup again.
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	pods, err = s.indexer.GetPodScores(s.T().Context(), renderReq, "", defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.Len(pods, 1, "expected one pod score after indexing")
	s.Greater(pods[s.Pod1IP], float64(0), "expected positive pod score")
	s.T().Logf("Chat completions flow score: %v", pods[s.Pod1IP])
}
