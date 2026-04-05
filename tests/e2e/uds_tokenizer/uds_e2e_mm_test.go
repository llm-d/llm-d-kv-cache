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
	"context"

	"k8s.io/apimachinery/pkg/util/sets"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

const (
	mmModelName = "Qwen/Qwen2-VL-2B-Instruct"
	// Two different COCO images from huggingface test fixtures.
	imageA = "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000039769.png"
	imageB = "https://raw.githubusercontent.com/huggingface/transformers/main/tests/fixtures/tests_samples/COCO/000000004016.png"
)

// switchToMMModel initializes the multimodal model.
// NewUdsTokenizer eagerly warms up the renderer, so no extra warmup needed.
func (s *UDSTokenizerSuite) switchToMMModel() {
	s.T().Helper()
	s.switchTokenizer(mmModelName)
}

// mmRenderResult holds the tokens and features from a multimodal RenderChat call.
type mmRenderResult struct {
	Tokens   []uint32
	Features *tokenization.MultiModalFeatures
}

// mmRenderChat sends a multimodal chat request with one image and returns the result.
func (s *UDSTokenizerSuite) mmRenderChat(imageURL, text string) *mmRenderResult {
	s.T().Helper()
	req := &types.RenderChatRequest{
		Conversation: []types.Conversation{{
			Role: "user",
			Content: types.Content{
				Structured: []types.ContentBlock{
					{Type: "image_url", ImageURL: types.ImageBlock{URL: imageURL}},
					{Type: "text", Text: text},
				},
			},
		}},
		AddGenerationPrompt: true,
	}
	tokens, features, err := s.tokenizer.RenderChat(req)
	s.Require().NoError(err, "multimodal RenderChat failed")
	s.Require().NotEmpty(tokens)
	return &mmRenderResult{Tokens: tokens, Features: features}
}

// TestMM_FeaturesReturned verifies that a multimodal request returns MM features
// with valid placeholder ranges, and that text-only requests return nil features.
func (s *UDSTokenizerSuite) TestMM_FeaturesReturned() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, "What is in this image?")

	s.Require().NotNil(result.Features, "multimodal request should return features")
	s.Require().Contains(result.Features.MMHashes, "image")
	s.Require().Contains(result.Features.MMPlaceholders, "image")

	hashes := result.Features.MMHashes["image"]
	placeholders := result.Features.MMPlaceholders["image"]
	s.Require().Len(hashes, 1, "one image should produce one hash")
	s.Require().Len(placeholders, 1, "one image should produce one placeholder range")
	s.Assert().NotEmpty(hashes[0], "image hash should not be empty")

	// Placeholder range should be within token bounds.
	ph := placeholders[0]
	s.Assert().GreaterOrEqual(ph.Offset, 0)
	s.Assert().Greater(ph.Length, 0)
	s.Assert().LessOrEqual(ph.Offset+ph.Length, len(result.Tokens),
		"placeholder [%d, %d) exceeds token count %d", ph.Offset, ph.Offset+ph.Length, len(result.Tokens))

	s.T().Logf("tokens=%d hash=%s placeholder=[%d,%d)",
		len(result.Tokens), hashes[0], ph.Offset, ph.Offset+ph.Length)

	// Text-only request should have no features.
	_, textFeatures, err := s.tokenizer.RenderChat(&types.RenderChatRequest{
		Conversation:        []types.Conversation{{Role: "user", Content: types.Content{Raw: "Tell me about cats"}}},
		AddGenerationPrompt: true,
	})
	s.Require().NoError(err)
	hasMMContent := textFeatures != nil &&
		(len(textFeatures.MMHashes) > 0 || len(textFeatures.MMPlaceholders) > 0)
	s.Assert().False(hasMMContent, "text-only request should not have MM features")
}

// TestMM_BlockFeatureAssignmentMatchesPlaceholders verifies that
// ComputeBlockExtraFeatures assigns features to exactly the blocks
// that overlap the placeholder range.
func (s *UDSTokenizerSuite) TestMM_BlockFeatureAssignmentMatchesPlaceholders() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, "What is in this image?")
	s.Require().NotNil(result.Features)

	blockFeatures := kvblock.ComputeBlockExtraFeatures(
		result.Features.MMHashes, result.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(result.Tokens),
	)

	numBlocks := len(result.Tokens) / s.tokenProcessorConfig.BlockSize
	s.Require().Len(blockFeatures, numBlocks)

	for mod, ranges := range result.Features.MMPlaceholders {
		for _, r := range ranges {
			for bi := 0; bi < numBlocks; bi++ {
				blockStart := bi * s.tokenProcessorConfig.BlockSize
				blockEnd := blockStart + s.tokenProcessorConfig.BlockSize
				overlaps := r.Offset < blockEnd && (r.Offset+r.Length) > blockStart
				hasFeat := blockFeatures[bi] != nil

				if overlaps {
					s.Assert().True(hasFeat,
						"block %d [%d,%d) overlaps %s [%d,%d) but has no features",
						bi, blockStart, blockEnd, mod, r.Offset, r.Offset+r.Length)
				} else {
					s.Assert().False(hasFeat,
						"block %d [%d,%d) does NOT overlap %s [%d,%d) but has features",
						bi, blockStart, blockEnd, mod, r.Offset, r.Offset+r.Length)
				}
			}
		}
	}
}

// TestMM_Determinism verifies that the same multimodal request produces
// identical MM hashes, tokens, and block keys across calls.
func (s *UDSTokenizerSuite) TestMM_Determinism() {
	s.switchToMMModel()

	r1 := s.mmRenderChat(imageA, "What is in this image?")
	r2 := s.mmRenderChat(imageA, "What is in this image?")

	s.Require().NotNil(r1.Features)
	s.Require().NotNil(r2.Features)

	s.Assert().Equal(r1.Tokens, r2.Tokens, "tokens should be identical")
	s.Assert().Equal(r1.Features.MMHashes, r2.Features.MMHashes, "MM hashes should be identical")

	// Block keys must match.
	bf1 := kvblock.ComputeBlockExtraFeatures(
		r1.Features.MMHashes, r1.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(r1.Tokens))
	bf2 := kvblock.ComputeBlockExtraFeatures(
		r2.Features.MMHashes, r2.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(r2.Tokens))

	keys1, err := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, r1.Tokens, mmModelName, bf1)
	s.Require().NoError(err)
	keys2, err := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, r2.Tokens, mmModelName, bf2)
	s.Require().NoError(err)

	s.Assert().Equal(keys1, keys2, "block keys should be identical across calls")
}

// TestMM_DifferentImagesProduceDifferentKeys verifies that two different images
// produce different content hashes and different block keys.
func (s *UDSTokenizerSuite) TestMM_DifferentImagesProduceDifferentKeys() {
	s.switchToMMModel()

	rA := s.mmRenderChat(imageA, "What is in this image?")
	rB := s.mmRenderChat(imageB, "What is in this image?")

	s.Require().NotNil(rA.Features)
	s.Require().NotNil(rB.Features)

	hashesA := rA.Features.MMHashes["image"]
	hashesB := rB.Features.MMHashes["image"]
	s.Require().NotEmpty(hashesA)
	s.Require().NotEmpty(hashesB)
	s.Assert().NotEqual(hashesA[0], hashesB[0],
		"different images should produce different content hashes")

	s.T().Logf("image A hash: %s", hashesA[0])
	s.T().Logf("image B hash: %s", hashesB[0])

	bfA := kvblock.ComputeBlockExtraFeatures(
		rA.Features.MMHashes, rA.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(rA.Tokens))
	bfB := kvblock.ComputeBlockExtraFeatures(
		rB.Features.MMHashes, rB.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(rB.Tokens))

	keysA, err := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, rA.Tokens, mmModelName, bfA)
	s.Require().NoError(err)
	keysB, err := s.tokenProcessor.TokensToKVBlockKeys(kvblock.EmptyBlockHash, rB.Tokens, mmModelName, bfB)
	s.Require().NoError(err)

	// At least some blocks should differ.
	minLen := len(keysA)
	if len(keysB) < minLen {
		minLen = len(keysB)
	}
	differ := 0
	for i := 0; i < minLen; i++ {
		if keysA[i] != keysB[i] {
			differ++
		}
	}
	s.Assert().Greater(differ, 0,
		"different images should produce at least some different block keys")
	s.T().Logf("%d/%d comparable blocks differ", differ, minLen)
}

// TestMM_TextBlocksBeforeImageUnaffected verifies that blocks before the
// image placeholder produce the same keys as a text-only computation.
func (s *UDSTokenizerSuite) TestMM_TextBlocksBeforeImageUnaffected() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, "What is in this image?")
	s.Require().NotNil(result.Features)

	blockFeatures := kvblock.ComputeBlockExtraFeatures(
		result.Features.MMHashes, result.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(result.Tokens),
	)

	keysWithMM, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, result.Tokens, mmModelName, blockFeatures)
	s.Require().NoError(err)

	keysTextOnly, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, result.Tokens, mmModelName, nil)
	s.Require().NoError(err)

	// Find first MM block.
	firstMM := -1
	for i, f := range blockFeatures {
		if f != nil {
			firstMM = i
			break
		}
	}

	if firstMM > 0 {
		for i := 0; i < firstMM; i++ {
			s.Assert().Equal(keysTextOnly[i], keysWithMM[i],
				"block %d before image should be identical to text-only", i)
		}
		s.T().Logf("blocks 0..%d (before image) match text-only", firstMM-1)
	} else {
		s.T().Log("image starts at block 0 — no pure-text prefix to compare")
	}
}

// TestMM_IndexLookupRoundTrip verifies the full ingestion→lookup cycle:
// ingest block keys with MM features, then look them up using request-path keys.
func (s *UDSTokenizerSuite) TestMM_IndexLookupRoundTrip() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, "Describe this image in detail")
	s.Require().NotNil(result.Features)

	blockFeatures := kvblock.ComputeBlockExtraFeatures(
		result.Features.MMHashes, result.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(result.Tokens),
	)

	// Compute request-path keys (what the indexer would compute from a new request).
	requestKeys, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, result.Tokens, mmModelName, blockFeatures)
	s.Require().NoError(err)
	s.Require().NotEmpty(requestKeys)

	// Simulate engine keys (different parent hash, same as real vLLM would produce).
	engineKeys, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.BlockHash(1), result.Tokens, mmModelName, blockFeatures)
	s.Require().NoError(err)

	// Add to index.
	podID := "mm-test-pod"
	err = s.kvBlockIndex.Add(s.T().Context(), engineKeys, requestKeys,
		[]kvblock.PodEntry{{PodIdentifier: podID, DeviceTier: "GPU"}})
	s.Require().NoError(err)

	// Look up using the same request keys — should find the pod.
	results, err := s.kvBlockIndex.Lookup(s.T().Context(), requestKeys, sets.New[string]())
	s.Require().NoError(err)

	found := 0
	for _, key := range requestKeys {
		if pods, ok := results[key]; ok && len(pods) > 0 {
			s.Assert().Equal(podID, pods[0].PodIdentifier)
			found++
		}
	}
	s.Assert().Equal(len(requestKeys), found,
		"all %d request keys should map to the pod, got %d", len(requestKeys), found)
	s.T().Logf("all %d blocks found in index via request-path keys", found)

	// Look up with a different image's keys — should NOT find the same pod.
	resultB := s.mmRenderChat(imageB, "Describe this image in detail")
	s.Require().NotNil(resultB.Features)

	bfB := kvblock.ComputeBlockExtraFeatures(
		resultB.Features.MMHashes, resultB.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(resultB.Tokens))
	keysB, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, resultB.Tokens, mmModelName, bfB)
	s.Require().NoError(err)

	resultsB, err := s.kvBlockIndex.Lookup(context.Background(), keysB, sets.New[string]())
	s.Require().NoError(err)

	// MM blocks should NOT match (different image hash → different keys).
	mmHits := 0
	for _, key := range keysB {
		if pods, ok := resultsB[key]; ok && len(pods) > 0 {
			mmHits++
		}
	}
	// Some text-only prefix blocks might match if prompts share a prefix,
	// but the MM blocks should not.
	s.Assert().Less(mmHits, len(keysB),
		"different image should not match all blocks (got %d/%d hits)", mmHits, len(keysB))
	s.T().Logf("different image: %d/%d blocks hit (expected < %d)", mmHits, len(keysB), len(keysB))
}
