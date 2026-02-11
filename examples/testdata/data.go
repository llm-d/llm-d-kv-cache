// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testdata

import (
	_ "embed"

	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

const (
	ModelName = "bert-base-uncased"
)

var RenderReq *types.RenderChatRequest = nil

//go:embed prompt.txt
var Prompt string

var PromptHashes = []uint64{
	3246512376769953277,
	2932514196368075983,
	6384763183060574933,
	13975137892230421288,
}
