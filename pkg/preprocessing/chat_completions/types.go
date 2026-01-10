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

package preprocessing

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

const nilString = "<nil>"

// RenderJinjaTemplateRequest is a structured representation of the fields we parse out of the v1/chat/completions
// request body. For detailed body fields, please refer to https://platform.openai.com/docs/api-reference/chat.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type RenderJinjaTemplateRequest struct {
	/* parameters from the official OpenAI chat-completions API */
	Messages []Message     `json:"messages,omitempty"`
	Tools    []interface{} `json:"tools,omitempty"`
	/* parameters from the HuggingFace transformers chat-templates API */
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}

func (r *RenderJinjaTemplateRequest) String() string {
	if r == nil {
		return nilString
	}

	messagesLen := 0
	for _, msg := range r.Messages {
		messagesLen += len(msg.Content.PlainText())
	}
	return fmt.Sprintf("{MessagesLength: %d}", messagesLen)
}

func (r *RenderJinjaTemplateRequest) DeepCopy() (*RenderJinjaTemplateRequest, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	var out RenderJinjaTemplateRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// Message represents a single message in a chat-completions request.
type Message struct {
	// Role is the message Role, optional values are 'user', 'assistant', ...
	Role string `json:"role,omitempty"`
	// Content defines text of this message
	Content Content `json:"content,omitempty"`
}

type Content struct {
	Raw        string
	Structured []ContentBlock
}

type ContentBlock struct {
	Type     string     `json:"type"`
	Text     string     `json:"text,omitempty"`
	ImageURL ImageBlock `json:"image_url,omitempty"`
}

type ImageBlock struct {
	URL string `json:"url,omitempty"`
}

// UnmarshalJSON allow use both format.
func (mc *Content) UnmarshalJSON(data []byte) error {
	// Raw format
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		mc.Raw = str
		return nil
	}

	// Block format
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		mc.Structured = blocks
		return nil
	}

	return errors.New("content format not supported")
}

func (mc Content) MarshalJSON() ([]byte, error) {
	if mc.Raw != "" {
		return json.Marshal(mc.Raw)
	}
	if mc.Structured != nil {
		return json.Marshal(mc.Structured)
	}
	return json.Marshal("")
}

func (mc Content) PlainText() string {
	if mc.Raw != "" {
		return mc.Raw
	}
	var sb strings.Builder
	for _, block := range mc.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

// RenderJinjaTemplateResponse represents the response from rendering a chat template.
type RenderJinjaTemplateResponse struct {
	RenderedChats     []string  `json:"rendered_chats"`
	GenerationIndices [][][]int `json:"generation_indices"`
}

// FetchChatTemplateRequest represents the request to fetch a chat template.
// This is needed if the fields are not set in the `RenderJinjaTemplateRequest`.
// When called, it will fetch the `chat_template` from the tokenizer.
// If the tokenizer is not present, it will be fetched from HuggingFace using
// the `token` if provided.
type FetchChatTemplateRequest struct {
	Model        string        `json:"model"`
	ChatTemplate string        `json:"chat_template,omitempty"`
	Tools        []interface{} `json:"tools,omitempty"`
	Revision     string        `json:"revision,omitempty"`
	Token        string        `json:"token,omitempty"`
	IsLocalPath  bool          `json:"is_local_path,omitempty"`
}

// FetchChatTemplateResponse represents the response from fetching a chat template.
type FetchChatTemplateResponse struct {
	ChatTemplate       string                 `json:"chat_template,omitempty"`
	ChatTemplateKWArgs map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}
