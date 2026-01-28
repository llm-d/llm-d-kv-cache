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

//nolint: gocritic // C and unsafe are considered dups by the linter.
import (
	"context"
	"encoding/json"
	"fmt"
	"unsafe"

	/*
		#cgo CFLAGS: -Wno-unused-variable
		#include "cgo_functions.h"
	*/
	"C"

	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type GetOrCreateTokenizerKeyRequest struct {
	IsLocal     bool   `json:"is_local,omitempty"`
	DownloadDir string `json:"download_dir,omitempty"`
	Model       string `json:"model"`
	Revision    string `json:"revision,omitempty"`
	Token       string `json:"token,omitempty"`
}

// Conversation represents a single message in a conversation.
// Content can be either a string (text-only) or a structured array of content blocks (multi-modality).
// This matches the OpenAI API format where content can be:
// - A string: "Hello, world!"
// - An array: [{"type": "text", "text": "Hello"}, {"type": "image_url", "image_url": {"url": "..."}}].
type Conversation struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // Can be string or []ContentBlock
}

// ContentBlock represents a single content block in a multi-modal message.
// This matches the OpenAI API format for structured content.
type ContentBlock struct {
	Type     string      `json:"type"`                // "text" or "image_url"
	Text     string      `json:"text,omitempty"`      // Only for type="text"
	ImageURL *ImageBlock `json:"image_url,omitempty"` // Only for type="image_url"
}

// ImageBlock represents an image in a content block.
type ImageBlock struct {
	URL string `json:"url"` // Can be HTTP URL or base64 data URL (data:image/...;base64,...)
}

// ApplyChatTemplateRequest represents the request to render a chat template.
type ApplyChatTemplateRequest struct {
	// The Python wrapper will handle converting this to a batched list if needed.
	Key                       string                 `json:"key"`
	Conversation              [][]Conversation       `json:"conversation"`
	Tools                     []interface{}          `json:"tools,omitempty"`
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}

// DeepCopy creates a deep copy of the ApplyChatTemplateRequest.
func (req *ApplyChatTemplateRequest) DeepCopy() (*ApplyChatTemplateRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out ApplyChatTemplateRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// ChatTemplatingProcessor is a processor that handles chat template rendering
// using a cached Python function. Once the Python interpreter is initialized,
// it caches the `vllm` function `apply_chat_template` for rendering
// chat templates. It also provides a method to fetch chat templates from the
// tokenizer or HuggingFace if the tokenizer is not present.
type ChatTemplatingProcessor struct{}

// NewChatTemplatingProcessor creates a new instance of ChatTemplatingProcessor.
func NewChatTemplatingProcessor() *ChatTemplatingProcessor {
	return &ChatTemplatingProcessor{}
}

// Initialize initializes the Python interpreter and caches the module.
func (w *ChatTemplatingProcessor) Initialize() error {
	// Initialize Python interpreter - C handles process-level tracking
	C.Py_InitializeGo()

	// Initialize chat template module - C handles module-level tracking
	result := C.Py_InitChatTemplateModule()
	if result != 0 {
		return fmt.Errorf("failed to initialize chat template module")
	}

	return nil
}

// Finalize finalizes the Python interpreter and cleans up the module.
func (w *ChatTemplatingProcessor) Finalize() {
	// Clean up the module first
	C.Py_CleanupChatTemplateModule()

	// Then finalize Python interpreter
	C.Py_FinalizeGo()
}

// GetOrCreateTokenizerKey returns the cache key for the tokenizer specified in the request.
func (w *ChatTemplatingProcessor) GetOrCreateTokenizerKey(
	ctx context.Context,
	req *GetOrCreateTokenizerKeyRequest,
) (string, error) {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("LoadTokenizer")
	if req == nil {
		traceLogger.Error(nil, "Received nil request")
		return "", fmt.Errorf("received nil request")
	}
	// Convert request to JSON
	reqJSON, err := json.Marshal(req)
	if err != nil {
		traceLogger.Error(err, "Failed to marshal request")
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}
	// Call the cached Python function
	cJSONString := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cJSONString))
	cResult := C.Py_CallGetOrCreateTokenizerKey(cJSONString)
	if cResult == nil {
		traceLogger.Error(nil, "C function returned nil")
		return "", fmt.Errorf("python get_or_create_tokenizer_key failed")
	}
	defer C.free(unsafe.Pointer(cResult))

	return C.GoString(cResult), nil
}

// ApplyChatTemplate renders a chat template using the cached Python function.
// It calls the Python `vllm` function `apply_chat_template` with the provided request.
func (w *ChatTemplatingProcessor) ApplyChatTemplate(ctx context.Context,
	req *ApplyChatTemplateRequest,
) (string, error) {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("ApplyChatTemplate")

	if req == nil {
		traceLogger.Error(nil, "Received nil request")
		return "", fmt.Errorf("received nil request")
	}

	reqJSON, err := json.Marshal(req)
	if err != nil {
		traceLogger.Error(err, "Failed to marshal request")
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}
	// Call the cached Python function
	cJSONString := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cJSONString))
	cResult := C.Py_CallApplyChatTemplate(cJSONString)
	if cResult == nil {
		traceLogger.Error(nil, "C function returned nil")
		return "", fmt.Errorf("python apply_chat_template failed")
	}
	defer C.free(unsafe.Pointer(cResult))

	return C.GoString(cResult), nil
}

// ClearCaches clears all caches for testing purposes.
func ClearCaches(ctx context.Context) error {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("clearCaches")

	// Call the C function
	cResult := C.Py_ClearCaches()
	if cResult == nil {
		traceLogger.Error(nil, "Failed to clear caches")
		return fmt.Errorf("failed to clear caches")
	}
	defer C.free(unsafe.Pointer(cResult))

	return nil
}
