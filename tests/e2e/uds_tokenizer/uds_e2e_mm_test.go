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
	"context"
	"fmt"
	"strings"

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

// ---------------------------------------------------------------------------
// Golden test cases — verify exact deterministic outputs for multimodal pipeline
// ---------------------------------------------------------------------------

// goldenFormatPlaceholderRanges formats placeholder ranges as Go source code.
func goldenFormatPlaceholderRanges(name string, ranges []kvblock.PlaceholderRange) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s = []kvblock.PlaceholderRange{", name)
	for i, r := range ranges {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "{Offset: %d, Length: %d}", r.Offset, r.Length)
	}
	b.WriteString("}")
	return b.String()
}

// goldenFormatStringSlice formats a []string as Go source code.
func goldenFormatStringSlice(name string, s []string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s = []string{", name)
	for i, v := range s {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%q", v)
	}
	b.WriteString("}")
	return b.String()
}

// Golden values for multimodal request with Qwen/Qwen2-VL-2B-Instruct.
// Uses imageA ("000000039769.png") with the prompt "What is in this image?"
// To regenerate: run TestGoldenMM_Tokenization and copy the logged output.
//
//nolint:gochecknoglobals // golden test fixtures
var (
	goldenMMPrompt = "What is in this image?"

	// Expected token IDs from multimodal RenderChat(imageA, goldenMMPrompt).
	// Structure: 15 prefix tokens + 391 placeholder tokens (151655) + 12 suffix tokens = 418 total.
	//nolint:dupword // repeated placeholder token IDs are expected
	goldenMMTokenIDs = []uint32{
		151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644,
		872, 198, 151652, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655,
		151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151655, 151653, 3838,
		374, 304, 419, 2168, 30, 151645, 198, 151644, 77091, 198,
	}

	// Expected MM content hashes (keyed by "image" modality).
	goldenMMHashes = []string{"f9a38a66df90291515d7ea4de3175b262c1da7668ad1cc32e3dbb98be86388ac"}

	// Expected placeholder ranges for the image modality.
	goldenMMPlaceholders = []kvblock.PlaceholderRange{{Offset: 15, Length: 391}}

	// Expected request keys from TokensToKVBlockKeys with MM extra features.
	goldenMMRequestKeys = []kvblock.BlockHash{
		14104998575452860231, 18378075856223045580, 11048751930103298547, 6129808321815669419,
		14326372486790561568, 1815176728854652040, 14583118626851175199, 11979531525341441311,
		1023528306349676745, 6953037652644231765, 1350392198618327489, 11778085822019237754,
		13316348388810461103, 13791542491503106800, 4837146218063722891, 6709173295186239608,
		1015430553157653266, 4433074773194310989, 11631483218740603807, 7278839243579545137,
		9706042548601691694, 15940053050553794773, 9384369000486015184, 13954739762694940680,
		13048533869259942367, 1365744505245295348, 3361686686510993187, 10616601766524037148,
		6364416232586540876, 3900643890518929093, 6156411931889457394, 17267008569216303193,
		2152398508896788794, 1048395817934769910, 13972100559240091313, 2983462691905975399,
		3391267063587469151, 313655964057035895, 11117621080894111455, 2591790146898088712,
		14188909595609494330, 13724723441206514095, 16221280722026278930, 16071827205786553608,
		16103790129283673690, 8261300976424493154, 7442296086261825643, 4301828717278229223,
		15894206806323655031, 13226680044231093608, 9531536856182053374, 4536474440236995267,
		2682947230332253577, 11399873370848950328, 12859529695331906656, 15598044574459391891,
		401266734069035553, 15462185159637555281, 13160261555784776306, 11141962100872942559,
		9364914506513864499, 6258567425678560463, 4503397171368799927, 4241344932566206282,
		401191476587858364, 11464321239991899209, 8761141402295810926, 9462299790155805339,
		9498688271434991305, 5744477554029469359, 6624989406610232503, 13833257604321085901,
		2234338363828797189, 12137051340783605760, 1215364924955133737, 4237116577282272079,
		7103548699163281060, 15459736333037470057, 13887568902698808679, 13040887960555373713,
		10830322734869969035, 15385119969436204211, 9987412763816791901, 16068607543006083497,
		181327409116310451, 8953992708400809643, 16220850766685735787, 6436017582899175878,
		3079309230528813266, 13468675238662324090, 1311341847949062688, 3683676030431607116,
		6343431959710464590, 6214904808533232703, 3524354708728900597, 6818295474426954080,
		13169489648165839466, 4755743376358206791, 13850117491056241728, 13733642666186928311,
		14801685921569033221, 13584187784946852980, 18378606778601056664, 9528532473271546174,
	}
)

// TestGoldenMM_Tokenization verifies that a fixed multimodal request produces
// the exact expected token IDs. If golden values are not yet set, the test
// logs the actual values in Go source format and skips.
func (s *UDSTokenizerSuite) TestGoldenMM_Tokenization() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, goldenMMPrompt)
	s.Require().NotNil(result.Features)

	if len(goldenMMTokenIDs) == 0 {
		s.T().Logf("GOLDEN VALUES NOT SET — copy the following into goldenMMTokenIDs:\n%s",
			goldenFormatUint32Slice("goldenMMTokenIDs", result.Tokens))
		s.T().Skip("golden MM token IDs not set yet; run once and copy the logged values")
	}

	s.Require().Equal(goldenMMTokenIDs, result.Tokens,
		"multimodal tokenization output changed — if intentional, update goldenMMTokenIDs")
	s.T().Logf("Golden MM tokenization verified: %d tokens match expected values", len(result.Tokens))
}

// TestGoldenMM_Features verifies that a fixed multimodal request produces
// the exact expected MM hashes and placeholder ranges.
func (s *UDSTokenizerSuite) TestGoldenMM_Features() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, goldenMMPrompt)
	s.Require().NotNil(result.Features)

	hashes := result.Features.MMHashes["image"]
	placeholders := result.Features.MMPlaceholders["image"]

	if len(goldenMMHashes) == 0 {
		s.T().Logf("GOLDEN VALUES NOT SET — copy the following:\n%s\n%s",
			goldenFormatStringSlice("goldenMMHashes", hashes),
			goldenFormatPlaceholderRanges("goldenMMPlaceholders", placeholders))
		s.T().Skip("golden MM features not set yet; run once and copy the logged values")
	}

	s.Require().Equal(goldenMMHashes, hashes,
		"MM hashes changed — if intentional, update goldenMMHashes")
	s.Require().Equal(goldenMMPlaceholders, placeholders,
		"MM placeholders changed — if intentional, update goldenMMPlaceholders")
	s.T().Logf("Golden MM features verified: hashes=%v placeholders=%v", hashes, placeholders)
}

// TestGoldenMM_BlockKeys verifies that computing block keys from fixed multimodal
// tokens and features produces the exact expected request keys.
func (s *UDSTokenizerSuite) TestGoldenMM_BlockKeys() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, goldenMMPrompt)
	s.Require().NotNil(result.Features)

	blockFeatures := kvblock.ComputeBlockExtraFeatures(
		result.Features.MMHashes, result.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(result.Tokens),
	)

	requestKeys, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, result.Tokens, mmModelName, blockFeatures)
	s.Require().NoError(err)
	s.Require().NotEmpty(requestKeys)

	if len(goldenMMRequestKeys) == 0 {
		s.T().Logf("GOLDEN VALUES NOT SET — copy the following into goldenMMRequestKeys:\n%s",
			goldenFormatBlockHashSlice("goldenMMRequestKeys", requestKeys))
		s.T().Skip("golden MM request keys not set yet; run once and copy the logged values")
	}

	s.Require().Equal(goldenMMRequestKeys, requestKeys,
		"MM block key computation changed — if intentional, update goldenMMRequestKeys")
	s.T().Logf("Golden MM block keys verified: %d keys match expected values", len(requestKeys))
}

// TestGoldenMM_Scoring verifies the full multimodal pipeline: tokenize → MM features →
// block keys → index → score. Uses deterministic inputs and verifies the exact score value.
// Uses ScoreTokens with pre-computed block features to avoid block-size mismatches
// between the test's token processor (blockSize=4) and the indexer's config.
func (s *UDSTokenizerSuite) TestGoldenMM_Scoring() {
	s.switchToMMModel()

	result := s.mmRenderChat(imageA, goldenMMPrompt)
	s.Require().NotNil(result.Features)

	blockFeatures := kvblock.ComputeBlockExtraFeatures(
		result.Features.MMHashes, result.Features.MMPlaceholders,
		s.tokenProcessorConfig.BlockSize, len(result.Tokens),
	)

	requestKeys, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.EmptyBlockHash, result.Tokens, mmModelName, blockFeatures)
	s.Require().NoError(err)

	engineKeys, err := s.tokenProcessor.TokensToKVBlockKeys(
		kvblock.BlockHash(1), result.Tokens, mmModelName, blockFeatures)
	s.Require().NoError(err)

	fakePodList := []string{s.Pod1IP}
	err = s.kvBlockIndex.Add(s.T().Context(), engineKeys, requestKeys,
		[]kvblock.PodEntry{{PodIdentifier: s.Pod1IP, DeviceTier: "gpu"}})
	s.Require().NoError(err)

	// Use ScoreTokens with pre-computed block features instead of GetPodScores,
	// because GetPodScores re-computes block features internally using the
	// indexer's config block size, which may differ from the test's blockSize=4.
	pods, err := s.indexer.ScoreTokens(s.T().Context(), result.Tokens, mmModelName, fakePodList, blockFeatures)
	s.Require().NoError(err)
	s.Require().Contains(pods, s.Pod1IP, "expected pod in scores")

	expectedScore := float64(len(requestKeys))
	s.Require().Equal(expectedScore, pods[s.Pod1IP],
		"score should equal number of matching block keys")
	s.T().Logf("Golden MM scoring: prompt=%q, blocks=%d, score=%.0f",
		goldenMMPrompt, len(requestKeys), pods[s.Pod1IP])
}
