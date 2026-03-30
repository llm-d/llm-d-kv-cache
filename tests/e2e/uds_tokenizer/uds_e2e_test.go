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
	"fmt"
	"strings"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

// ---------------------------------------------------------------------------
// Tokenization-only tests
// ---------------------------------------------------------------------------

// TestTokenize verifies that the UDS tokenizer produces non-empty, deterministic results.
func (s *UDSTokenizerSuite) TestTokenize() {
	prompt := "What is the capital of France?"

	tokens1, _, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens1, "token IDs should not be empty")
	s.T().Logf("Tokenized %d tokens for prompt", len(tokens1))

	// Tokenize the same prompt again — results must be identical (determinism).
	tokens2, _, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().Equal(tokens1, tokens2, "repeated tokenization should be deterministic (token IDs)")
}

// TestTokenizeWithSpecialTokens verifies that Encode(prompt, true) includes special tokens
// and Encode(prompt, false) does not.
// Uses TinyLlama (decode-only) which adds <s> BOS token for strict greater-than comparison.
func (s *UDSTokenizerSuite) TestTokenizeWithSpecialTokens() {
	// Switch to TinyLlama — a decode-only model that adds BOS (<s>) special token
	s.switchTokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

	prompt := "Hello world"

	tokensWithSpecial, _, err := s.tokenizer.Encode(prompt, true)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokensWithSpecial)

	tokensWithoutSpecial, _, err := s.tokenizer.Encode(prompt, false)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokensWithoutSpecial)

	// TinyLlama adds <s> (BOS) at the start when add_special_tokens=true.
	// So tokens with special tokens should always be strictly greater.
	s.Require().Greater(len(tokensWithSpecial), len(tokensWithoutSpecial),
		"encoding with special tokens should produce more tokens (TinyLlama adds BOS)")

	// Verify BOS token ID
	bosTokenID := uint32(1) // <s>
	s.Require().Equal(bosTokenID, tokensWithSpecial[0], "first token should be <s> (1)")

	s.T().Logf("Tokens with special: %d, without special: %d", len(tokensWithSpecial), len(tokensWithoutSpecial))
}

// TestRenderChatTemplate tests rendering a multi-turn conversation via the
// model's tokenizer chat template.
func (s *UDSTokenizerSuite) TestRenderChatTemplate() {
	conversation := []types.Conversation{
		{Role: "user", Content: types.Content{Raw: "What is machine learning?"}},
		{Role: "assistant", Content: types.Content{Raw: "Machine learning is a subset of AI."}},
		{Role: "user", Content: types.Content{Raw: "Give me an example."}},
	}

	renderReq := &types.RenderChatRequest{
		Conversation: conversation,
	}

	tokens, _, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err, "RenderChat should succeed")
	s.Require().NotEmpty(tokens, "rendered tokens should not be empty")
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
// Proto field contract tests
// ---------------------------------------------------------------------------

// TestAddGenerationPromptContract verifies that add_generation_prompt=false
// is correctly transmitted to the Python server and produces different output
// than add_generation_prompt=true. This guards against proto3 default value
// issues where false might be silently converted to true (see #432).
func (s *UDSTokenizerSuite) TestAddGenerationPromptContract() {
	conversation := []types.Conversation{
		{Role: "user", Content: types.Content{Raw: "What is the capital of France?"}},
	}

	reqTrue := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: true,
	}
	tokensTrue, _, err := s.tokenizer.RenderChat(reqTrue)
	s.Require().NoError(err, "RenderChat with AddGenerationPrompt=true should succeed")
	s.Require().NotEmpty(tokensTrue)

	reqFalse := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: false,
	}
	tokensFalse, _, err := s.tokenizer.RenderChat(reqFalse)
	s.Require().NoError(err, "RenderChat with AddGenerationPrompt=false should succeed")
	s.Require().NotEmpty(tokensFalse)

	s.Require().NotEqual(tokensTrue, tokensFalse,
		"add_generation_prompt=true and false must produce different tokens")

	// add_generation_prompt=true appends the assistant turn marker,
	// so it should produce strictly more tokens.
	s.Require().Greater(len(tokensTrue), len(tokensFalse),
		"add_generation_prompt=true should produce more tokens than false")

	// The false output should be a prefix of the true output.
	s.Require().Equal(tokensTrue[:len(tokensFalse)], tokensFalse,
		"tokens with add_generation_prompt=false should be a prefix of true")

	s.T().Logf("AddGenerationPrompt contract: true=%d tokens, false=%d tokens", len(tokensTrue), len(tokensFalse))
}

// TestAddGenerationPromptDefault verifies behavior when AddGenerationPrompt
// is not explicitly set (Go zero value = false). The proto field is optional,
// so the Python server should use its own default (true) when the field is absent.
func (s *UDSTokenizerSuite) TestAddGenerationPromptDefault() {
	conversation := []types.Conversation{
		{Role: "user", Content: types.Content{Raw: "What is the capital of France?"}},
	}

	// AddGenerationPrompt not set — Go zero value is false.
	// Since the proto field is optional, the Python server should default to true.
	reqDefault := &types.RenderChatRequest{
		Conversation: conversation,
	}
	tokensDefault, _, err := s.tokenizer.RenderChat(reqDefault)
	s.Require().NoError(err, "RenderChat with default AddGenerationPrompt should succeed")
	s.Require().NotEmpty(tokensDefault)

	reqExplicitTrue := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: true,
	}
	tokensExplicitTrue, _, err := s.tokenizer.RenderChat(reqExplicitTrue)
	s.Require().NoError(err)

	// When AddGenerationPrompt is false (zero value), the Go client sends
	// &false via the optional proto field. The Python server sees HasField=true
	// and uses false. Verify this produces fewer tokens than explicit true.
	reqExplicitFalse := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: false,
	}
	tokensExplicitFalse, _, err := s.tokenizer.RenderChat(reqExplicitFalse)
	s.Require().NoError(err)

	s.T().Logf("AddGenerationPrompt default: default=%d tokens, explicit-true=%d tokens, explicit-false=%d tokens",
		len(tokensDefault), len(tokensExplicitTrue), len(tokensExplicitFalse))

	// The Go client always sends &renderReq.AddGenerationPrompt (a pointer),
	// so default (false) behaves the same as explicit false.
	s.Require().Equal(tokensExplicitFalse, tokensDefault,
		"default (zero value false) should behave the same as explicit false since Go always sends the field")
}

// TestContinueFinalMessageContract verifies that continue_final_message=true
// is correctly transmitted and produces different output than false.
// When continue_final_message=true, the template should not close the last
// assistant turn, producing different tokenization.
func (s *UDSTokenizerSuite) TestContinueFinalMessageContract() {
	conversation := []types.Conversation{
		{Role: "user", Content: types.Content{Raw: "What is machine learning?"}},
		{Role: "assistant", Content: types.Content{Raw: "Machine learning is"}},
	}

	reqContinue := &types.RenderChatRequest{
		Conversation:         conversation,
		ContinueFinalMessage: true,
		AddGenerationPrompt:  false,
	}
	tokensContinue, _, err := s.tokenizer.RenderChat(reqContinue)
	s.Require().NoError(err, "RenderChat with ContinueFinalMessage=true should succeed")
	s.Require().NotEmpty(tokensContinue)

	reqNoContinue := &types.RenderChatRequest{
		Conversation:         conversation,
		ContinueFinalMessage: false,
		AddGenerationPrompt:  false,
	}
	tokensNoContinue, _, err := s.tokenizer.RenderChat(reqNoContinue)
	s.Require().NoError(err, "RenderChat with ContinueFinalMessage=false should succeed")
	s.Require().NotEmpty(tokensNoContinue)

	s.Require().NotEqual(tokensContinue, tokensNoContinue,
		"continue_final_message=true and false must produce different tokens when last message is from assistant")

	s.T().Logf("ContinueFinalMessage contract: continue=%d tokens, no-continue=%d tokens",
		len(tokensContinue), len(tokensNoContinue))
}

// TestChatTemplateOverride verifies that the ChatTemplate field is correctly
// transmitted to the Python server. When a per-request ChatTemplate is sent
// but trust_request_chat_template is not enabled on the server, vLLM rejects
// it with an error. This test validates that the field is actually received
// (not silently dropped), and that an empty ChatTemplate produces the same
// result as an unset one (model default is used).
func (s *UDSTokenizerSuite) TestChatTemplateOverride() {
	conversation := []types.Conversation{
		{Role: "user", Content: types.Content{Raw: "What is the capital of France?"}},
	}

	// A non-empty ChatTemplate should be rejected by the server (vLLM's
	// secure default requires trust_request_chat_template to be enabled).
	// The fact that an error is returned proves the field was transmitted.
	reqCustom := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: true,
		ChatTemplate:        `{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}`,
	}
	_, _, err := s.tokenizer.RenderChat(reqCustom)
	s.Require().Error(err, "custom ChatTemplate should be rejected when trust is not enabled")
	s.T().Logf("ChatTemplate override correctly rejected: %v", err)

	// An empty ChatTemplate should be treated as "use model default" and succeed.
	reqDefault := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: true,
	}
	tokensDefault, _, err := s.tokenizer.RenderChat(reqDefault)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokensDefault)

	reqEmptyTemplate := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: true,
		ChatTemplate:        "",
	}
	tokensEmpty, _, err := s.tokenizer.RenderChat(reqEmptyTemplate)
	s.Require().NoError(err)
	s.Require().Equal(tokensDefault, tokensEmpty,
		"empty ChatTemplate should produce the same tokens as unset (model default)")

	s.T().Logf("ChatTemplate override: default=%d tokens, empty=%d tokens", len(tokensDefault), len(tokensEmpty))
}

// TestChatTemplateKwargsPassthrough verifies that ChatTemplateKWArgs are
// correctly serialized to JSON and passed through to the Python server
// without error.
func (s *UDSTokenizerSuite) TestChatTemplateKwargsPassthrough() {
	conversation := []types.Conversation{
		{Role: "user", Content: types.Content{Raw: "Hello"}},
	}

	reqWithKwargs := &types.RenderChatRequest{
		Conversation:        conversation,
		AddGenerationPrompt: true,
		ChatTemplateKWArgs: map[string]interface{}{
			"custom_key": "custom_value",
		},
	}
	tokens, _, err := s.tokenizer.RenderChat(reqWithKwargs)
	s.Require().NoError(err, "RenderChat with ChatTemplateKWArgs should succeed without error")
	s.Require().NotEmpty(tokens)
	s.T().Logf("ChatTemplateKwargs passthrough: %d tokens", len(tokens))
}

// ---------------------------------------------------------------------------
// Golden test cases — verify exact deterministic outputs at each pipeline stage
// ---------------------------------------------------------------------------

// goldenFormatUint32Slice formats a []uint32 as Go source code for easy copy-paste.
func goldenFormatUint32Slice(name string, s []uint32) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s = []uint32{", name)
	for i, v := range s {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d", v)
	}
	b.WriteString("}")
	return b.String()
}

// goldenFormatBlockHashSlice formats a []BlockHash as Go source code.
func goldenFormatBlockHashSlice(name string, s []kvblock.BlockHash) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s = []kvblock.BlockHash{", name)
	for i, v := range s {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%d", v)
	}
	b.WriteString("}")
	return b.String()
}

// Golden values for "What is the capital of France?" with ibm-granite/granite-3.1-8b-instruct.
// To regenerate: run TestGoldenTokenization and copy the logged output.
//
//nolint:gochecknoglobals // golden test fixtures
var (
	goldenPrompt = "What is the capital of France?"

	// Expected token IDs from Render(goldenPrompt).
	goldenTokenIDs = []uint32{8197, 438, 322, 18926, 432, 45600, 49}

	// Expected request keys from TokensToKVBlockKeys(EmptyBlockHash, goldenTokenIDs, model, nil).
	goldenRequestKeys = []kvblock.BlockHash{1334984470577408192}
)

// Golden values for a chat conversation with ibm-granite/granite-3.1-8b-instruct.
// NOTE: The conversation starts with a system message to provide a fixed system prompt.
// Without it, Granite's chat template injects the current date via strftime_now(),
// making token output non-deterministic across days.
//
//nolint:gochecknoglobals // golden test fixtures
var (
	goldenChatConversation = []types.Conversation{
		{Role: "system", Content: types.Content{Raw: "You are a helpful assistant."}},
		{Role: "user", Content: types.Content{Raw: "What is machine learning?"}},
		{Role: "assistant", Content: types.Content{Raw: "Machine learning is a subset of AI."}},
		{Role: "user", Content: types.Content{Raw: "Give me an example."}},
	}

	// Expected token IDs from RenderChat(goldenChatConversation).
	// To regenerate: run TestGoldenChatTokenization and copy the logged output.
	goldenChatTokenIDs = []uint32{
		49152, 2946, 49153, 4282, 884, 312, 17247, 47330, 32, 0, 203,
		49152, 496, 49153, 8197, 438, 6652, 9608, 49, 0, 203,
		49152, 17594, 49153, 7090, 9608, 438, 312, 17272, 432, 19551, 32, 0, 203,
		49152, 496, 49153, 36780, 597, 600, 2280, 32, 0, 203,
	}
)

// TestGoldenTokenization verifies that tokenizing a fixed prompt produces
// the exact expected token IDs. If golden values are not yet set, the test
// logs the actual values in Go source format and skips.
func (s *UDSTokenizerSuite) TestGoldenTokenization() {
	tokens, _, err := s.tokenizer.Render(goldenPrompt)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)

	if len(goldenTokenIDs) == 0 {
		s.T().Logf("GOLDEN VALUES NOT SET — copy the following into goldenTokenIDs:\n%s",
			goldenFormatUint32Slice("goldenTokenIDs", tokens))
		s.T().Skip("golden token IDs not set yet; run once and copy the logged values")
	}

	s.Require().Equal(goldenTokenIDs, tokens,
		"tokenization output changed — if intentional, update goldenTokenIDs")
	s.T().Logf("Golden tokenization verified: %d tokens match expected values", len(tokens))
}

// TestGoldenBlockKeys verifies that computing block keys from fixed tokens
// produces the exact expected request keys.
func (s *UDSTokenizerSuite) TestGoldenBlockKeys() {
	tokens, _, err := s.tokenizer.Render(goldenPrompt)
	s.Require().NoError(err)

	requestKeys, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, tokens, defaultModelName, nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(requestKeys)

	if len(goldenRequestKeys) == 0 {
		s.T().Logf("GOLDEN VALUES NOT SET — copy the following into goldenRequestKeys:\n%s",
			goldenFormatBlockHashSlice("goldenRequestKeys", requestKeys))
		s.T().Skip("golden request keys not set yet; run once and copy the logged values")
	}

	s.Require().Equal(goldenRequestKeys, requestKeys,
		"block key computation changed — if intentional, update goldenRequestKeys")
	s.T().Logf("Golden block keys verified: %d keys match expected values", len(requestKeys))
}

// TestGoldenChatTokenization verifies that rendering a fixed chat conversation
// produces the exact expected token IDs.
func (s *UDSTokenizerSuite) TestGoldenChatTokenization() {
	renderReq := &types.RenderChatRequest{
		Conversation: goldenChatConversation,
	}
	tokens, _, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)

	if len(goldenChatTokenIDs) == 0 {
		s.T().Logf("GOLDEN VALUES NOT SET — copy the following into goldenChatTokenIDs:\n%s",
			goldenFormatUint32Slice("goldenChatTokenIDs", tokens))
		s.T().Skip("golden chat token IDs not set yet; run once and copy the logged values")
	}

	s.Require().Equal(goldenChatTokenIDs, tokens,
		"chat tokenization output changed — if intentional, update goldenChatTokenIDs")
	s.T().Logf("Golden chat tokenization verified: %d tokens match expected values", len(tokens))
}

// TestGoldenScoring verifies the full pipeline: tokenize → block keys → index → score.
// Uses deterministic inputs and verifies the exact score value.
func (s *UDSTokenizerSuite) TestGoldenScoring() {
	tokens, _, err := s.tokenizer.Render(goldenPrompt)
	s.Require().NoError(err)

	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(tokens)
	fakePodList := []string{s.Pod1IP}
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	pods, err := s.indexer.GetPodScores(s.T().Context(), nil, goldenPrompt, defaultModelName, fakePodList)
	s.Require().NoError(err)
	s.Require().Contains(pods, s.Pod1IP, "expected pod in scores")

	expectedScore := float64(len(requestKeys))
	s.Require().Equal(expectedScore, pods[s.Pod1IP],
		"score should equal number of matching block keys")
	s.T().Logf("Golden scoring: prompt=%q, blocks=%d, score=%.0f", goldenPrompt, len(requestKeys), pods[s.Pod1IP])
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
		{Role: "system", Content: types.Content{Raw: "You are a helpful AI assistant."}},
		{Role: "user", Content: types.Content{Raw: "What is the capital of France?"}},
		{Role: "assistant", Content: types.Content{Raw: "The capital of France is Paris."}},
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

// ---------------------------------------------------------------------------
// ScoreTokens tests
// ---------------------------------------------------------------------------

// TestScoreTokensCacheHit tokenizes a prompt externally via UDS,
// adds block keys to the index, then calls ScoreTokens with the
// pre-tokenized input and verifies positive scores.
func (s *UDSTokenizerSuite) TestScoreTokensCacheHit() {
	//nolint:lll // long prompt
	prompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	fakePodList := []string{s.Pod1IP}

	tokens, _, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)

	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(tokens)
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	pods, err := s.indexer.ScoreTokens(s.T().Context(), tokens, defaultModelName, fakePodList, nil)
	s.Require().NoError(err)
	s.T().Logf("ScoreTokens scores: %+v", pods)
	s.Len(pods, len(fakePodList), "expected pod scores length to match candidate pods")
	s.Greater(pods[s.Pod1IP], 1.0, "expected positive pod score")
}

// TestScoreTokensCacheMiss calls ScoreTokens for tokens
// that have no index entries and verifies empty scores.
func (s *UDSTokenizerSuite) TestScoreTokensCacheMiss() {
	prompt := "What is the capital of France?"
	fakePodList := []string{s.Pod1IP}

	tokens, _, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)

	pods, err := s.indexer.ScoreTokens(s.T().Context(), tokens, defaultModelName, fakePodList, nil)
	s.Require().NoError(err)
	s.T().Logf("ScoreTokens scores: %+v", pods)
	s.Empty(pods, "expected no pod scores since no keys were added to the index")
}

// TestScoreTokensConsistentWithGetPodScores tokenizes a prompt,
// indexes the block keys, then calls both GetPodScores and
// ScoreTokens and verifies they return identical scores.
func (s *UDSTokenizerSuite) TestScoreTokensConsistentWithGetPodScores() {
	//nolint:lll // long prompt
	prompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
	fakePodList := []string{s.Pod1IP}

	tokens, _, err := s.tokenizer.Render(prompt)
	s.Require().NoError(err)
	s.Require().NotEmpty(tokens)

	engineKeys, requestKeys := s.promptToEngineAndRequestKeys(tokens)
	s.addEntriesToIndex(engineKeys, requestKeys, fakePodList)

	scoresFromPrompt, err := s.indexer.GetPodScores(s.T().Context(), nil, prompt, defaultModelName, fakePodList)
	s.Require().NoError(err)

	scoresFromTokens, err := s.indexer.ScoreTokens(s.T().Context(), tokens, defaultModelName, fakePodList, nil)
	s.Require().NoError(err)

	s.Equal(scoresFromPrompt, scoresFromTokens,
		"GetPodScores and ScoreTokens should return identical scores for the same input")
	s.T().Logf("Both methods returned: %+v", scoresFromTokens)
}

// TestScoreTokensPrefixReduction indexes a full prompt's block keys,
// then queries with progressively shorter token prefixes via
// ScoreTokens, verifying partial-match scoring.
func (s *UDSTokenizerSuite) TestScoreTokensPrefixReduction() {
	//nolint:lll // long prompt
	fullPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat."
	shortPrompt := "lorem ipsum dolor sit amet, consectetur adipiscing elit."

	fullTokens, _, err := s.tokenizer.Render(fullPrompt)
	s.Require().NoError(err)

	fullEngineKeys, fullRequestKeys := s.promptToEngineAndRequestKeys(fullTokens)
	fakePodList := []string{s.Pod1IP}
	s.addEntriesToIndex(fullEngineKeys, fullRequestKeys, fakePodList)

	// Query with the short prompt's tokens — should produce a partial match.
	shortTokens, _, err := s.tokenizer.Render(shortPrompt)
	s.Require().NoError(err)

	pods, err := s.indexer.ScoreTokens(s.T().Context(), shortTokens, defaultModelName, fakePodList, nil)
	s.Require().NoError(err)
	s.Len(pods, len(fakePodList), "expected pod scores for short token prefix")
	s.T().Logf("Short prefix scores: %+v", pods)

	_, shortRequestKeys := s.promptToEngineAndRequestKeys(shortTokens)
	s.Equal(int(pods[s.Pod1IP]), len(shortRequestKeys),
		"all short-prefix block keys should have been indexed")
}
