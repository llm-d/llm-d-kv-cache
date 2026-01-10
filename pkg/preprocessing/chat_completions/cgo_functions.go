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

// ChatTemplatingProcessor is a processor that handles chat template rendering
// using a cached Python function. Once the Python interpreter is initialized,
// it caches the `transformers` function `render_jinja_template` for rendering
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

// RenderChatTemplate renders a chat template using the cached Python function.
// It calls the Python `transformers` function `render_jinja_template` with the provided request.
//
//nolint:gocritic // hugeParam: req is passed by value intentionally for immutability, but can consider using pointer.
func (w *ChatTemplatingProcessor) RenderChatTemplate(ctx context.Context,
	req *RenderJinjaTemplateRequest,
) (*RenderJinjaTemplateResponse, error) {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("RenderChatTemplate")
	if req == nil {
		traceLogger.Error(nil, "Received nil request")
		return nil, fmt.Errorf("received nil request")
	}

	// Convert request to JSON
	reqJSON, err := json.Marshal(req)
	if err != nil {
		traceLogger.Error(err, "Failed to marshal request")
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	// Call the cached Python function
	cJSONString := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cJSONString))
	cResult := C.Py_CallRenderJinjaTemplate(cJSONString)
	if cResult == nil {
		traceLogger.Error(nil, "C function returned nil")
		return nil, fmt.Errorf("python render_jinja_template failed")
	}
	defer C.free(unsafe.Pointer(cResult))
	resultJSON := C.GoString(cResult)

	// Parse the response
	var response RenderJinjaTemplateResponse
	if err := json.Unmarshal([]byte(resultJSON), &response); err != nil {
		traceLogger.Error(err, "Failed to unmarshal response")
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return &response, nil
}

// FetchChatTemplate fetches the model chat template using the cached Python function.
//
//nolint:gocritic // hugeParam: req is passed by value intentionally for immutability, but can consider using pointer.
func (w *ChatTemplatingProcessor) FetchChatTemplate(
	ctx context.Context,
	req FetchChatTemplateRequest,
) (string, map[string]interface{}, error) {
	traceLogger := log.FromContext(ctx).V(logging.TRACE).WithName("FetchChatTemplate")

	// Convert request to JSON
	reqJSON, err := json.Marshal(req)
	if err != nil {
		traceLogger.Error(err, "Failed to marshal request")
		return "", nil, fmt.Errorf("failed to marshal request: %w", err)
	}
	// Call the cached Python function
	cJSONString := C.CString(string(reqJSON))
	defer C.free(unsafe.Pointer(cJSONString))
	cResult := C.Py_CallGetModelChatTemplate(cJSONString)
	if cResult == nil {
		traceLogger.Error(nil, "C function returned nil")
		return "", nil, fmt.Errorf("python get_model_chat_template failed")
	}
	defer C.free(unsafe.Pointer(cResult))
	resultJSON := C.GoString(cResult)

	// Parse the response
	var response FetchChatTemplateResponse
	if err := json.Unmarshal([]byte(resultJSON), &response); err != nil {
		traceLogger.Error(err, "Failed to unmarshal response")
		return "", nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return response.ChatTemplate, response.ChatTemplateKWArgs, nil
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
