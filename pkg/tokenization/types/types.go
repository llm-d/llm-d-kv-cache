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

package types

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
)

// ImageBlock represents the image_url field in a multimodal content block.
type ImageBlock struct {
	URL string `json:"url,omitempty"`
}

// AudioBlock represents the input_audio field in a multimodal content block.
type AudioBlock struct {
	Data   string `json:"data,omitempty"`
	Format string `json:"format,omitempty"`
}

// AudioURLBlock represents the audio_url field in a multimodal content block.
type AudioURLBlock struct {
	URL string `json:"url,omitempty"`
}

// VideoBlock represents the video_url field in a multimodal content block.
type VideoBlock struct {
	URL string `json:"url,omitempty"`
}

// ContentBlock represents a single part of a multimodal message.
type ContentBlock struct {
	Type       string        `json:"type"`
	Text       string        `json:"text,omitempty"`
	ImageURL   ImageBlock    `json:"image_url,omitempty"`
	InputAudio AudioBlock    `json:"input_audio,omitempty"`
	AudioURL   AudioURLBlock `json:"audio_url,omitempty"`
	VideoURL   VideoBlock    `json:"video_url,omitempty"`
}

// Content holds a message's content — either plain text or a list of multimodal blocks.
type Content struct {
	Raw        string
	Structured []ContentBlock
}

// UnmarshalJSON handles both the plain-string and block-list formats.
func (c *Content) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		c.Raw = s
		return nil
	}
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		c.Structured = blocks
		return nil
	}
	return errors.New("content must be a string or array of content blocks")
}

// MarshalJSON serialises back to the original format.
func (c Content) MarshalJSON() ([]byte, error) {
	if c.Raw != "" {
		return json.Marshal(c.Raw)
	}
	if len(c.Structured) > 0 {
		return json.Marshal(c.Structured)
	}
	return json.Marshal("")
}

// PlainText returns the plain text content, concatenating text blocks for multimodal messages.
func (c Content) PlainText() string {
	if c.Raw != "" {
		return c.Raw
	}
	var sb strings.Builder
	for _, block := range c.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

// Conversation represents a single message in a conversation.
type Conversation struct {
	Role    string  `json:"role"`
	Content Content `json:"content"`
}

// RenderChatRequest represents the request to render a chat template.
type RenderChatRequest struct {
	// The Python wrapper will handle converting this to a batched list if needed.
	Key                       string                 `json:"key"`
	Conversation              []Conversation         `json:"conversation"`
	Tools                     []interface{}          `json:"tools,omitempty"`
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	TruncatePromptTokens      *int                   `json:"truncate_prompt_tokens,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}

type RenderRequest struct {
	Key              string `json:"key"`
	Text             string `json:"text"`
	AddSpecialTokens bool   `json:"add_special_tokens,omitempty"`
}

// RenderResponsesRequest represents the request to render a Responses API input.
type RenderResponsesRequest struct {
	Key          string `json:"key"`
	Input        any    `json:"input"`
	Instructions any    `json:"instructions,omitempty"`
	Tools        any    `json:"tools,omitempty"`
}

// Offset represents a character offset range with [start, end] indices.
type Offset [2]uint

// MultiModalFeatures carries the multimodal metadata returned alongside
// rendered token IDs. The wire format (lowercase snake_case keys, plain
// {offset, length} placeholder dicts) is set by the Python tokenizer wrapper.
// Empty maps indicate a text-only input.
type MultiModalFeatures struct {
	MMHashes       map[string][]string                   `json:"mm_hashes,omitempty"`
	MMPlaceholders map[string][]kvblock.PlaceholderRange `json:"mm_placeholders,omitempty"`
}

// IsEmpty reports whether the features hold no multimodal data.
func (f *MultiModalFeatures) IsEmpty() bool {
	return f == nil || (len(f.MMHashes) == 0 && len(f.MMPlaceholders) == 0)
}

type RenderResponse struct {
	TokenIDs       []uint32 `json:"input_ids"`
	OffsetMappings []Offset `json:"offset_mapping"`
	// MultiModalFeatures is embedded so the wrapper's top-level mm_hashes /
	// mm_placeholders keys unmarshal directly without an extra nesting level.
	MultiModalFeatures
}

// MMFeatures returns a pointer to the multimodal portion of the response, or
// nil for text-only outputs. The returned value shares the maps with the
// receiver (no deep copy).
func (r *RenderResponse) MMFeatures() *MultiModalFeatures {
	if r.IsEmpty() {
		return nil
	}
	f := r.MultiModalFeatures
	return &f
}

// DeepCopy creates a deep copy of the RenderChatRequest.
func (req *RenderChatRequest) DeepCopy() (*RenderChatRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderChatRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// DeepCopy creates a deep copy of the RenderRequest.
func (req *RenderRequest) DeepCopy() (*RenderRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// DeepCopy creates a deep copy of the RenderResponsesRequest.
func (req *RenderResponsesRequest) DeepCopy() (*RenderResponsesRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderResponsesRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}
