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

package kvcache

import (
	"context"

	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

// TokenizersPool abstracts the tokenization pool for testability/mocking.
type TokenizersPool interface {
	Tokenize(renderReq *types.RenderChatRequest, prompt string) ([]uint32, *tokenization.MultiModalFeatures)
	Run(ctx context.Context)
	SetTokenizer(tokenizer tokenization.Tokenizer, modelName string)
}
