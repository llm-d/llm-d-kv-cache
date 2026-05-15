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

package tokenization

import (
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

// MultiModalFeatures holds multimodal metadata produced by the tokenizer.
// Decoupled from proto types so callers don't depend on generated code.
// Defined in tokenization/types so that lower-level packages (e.g. the CGO
// wrapper in preprocessing) can return it without an import cycle.
type MultiModalFeatures = types.MultiModalFeatures

// Tokenizer interface defines the methods for tokenization.
type Tokenizer interface {
	RenderResponses(*types.RenderResponsesRequest) ([]uint32, *MultiModalFeatures, error)
	RenderChat(*types.RenderChatRequest) ([]uint32, *MultiModalFeatures, error)
	Render(string) ([]uint32, []types.Offset, error)
	Type() string
}

// TokenizerOptions holds common tokenizer configuration options shared by both
// HuggingFace and Local tokenizer configurations.
type TokenizerOptions struct {
	// Tokenizer is the tokenizer to use. If not specified, defaults to the model path.
	Tokenizer string `json:"tokenizer,omitempty"`
	// TokenizerMode specifies the tokenizer mode. Options:
	//   - "auto" (default): use mistral_common for Mistral models if available, otherwise "hf"
	//   - "hf": use the fast tokenizer if available
	//   - "slow": always use the slow tokenizer
	//   - "mistral": always use the tokenizer from mistral_common
	//   - "deepseek_v32": always use the tokenizer from deepseek_v32
	TokenizerMode string `json:"tokenizerMode,omitempty"`
	// TokenizerRevision is the revision of the tokenizer to use.
	TokenizerRevision string `json:"tokenizerRevision,omitempty"`
}

// DefaultTokenizerOptions returns the default tokenizer options.
func DefaultTokenizerOptions() TokenizerOptions {
	return TokenizerOptions{
		Tokenizer:         "",
		TokenizerMode:     "auto",
		TokenizerRevision: "",
	}
}
