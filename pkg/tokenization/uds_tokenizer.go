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
	"context"
	"fmt"
	"time"

	tokenizerpb "github.com/llm-d/llm-d-kv-cache/api/tokenizerpb"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// UdsTokenizerConfig represents the configuration for the UDS-based tokenizer,
// including the socket file path and other settings or TCP address (for testing only).
type UdsTokenizerConfig struct {
	SocketFile         string `json:"socketFile"`         // Path to the UDS socket file
	HuggingFaceToken   string `json:"huggingFaceToken"`   // Hugging Face token for private models
	TokenizersCacheDir string `json:"tokenizersCacheDir"` // Directory for caching tokenizers
	UseTCP             bool   `json:"useTCP"`             // If true, use TCP instead of UDS (for testing only, default: false)
}

func (cfg *UdsTokenizerConfig) IsEnabled() bool {
	return cfg != nil && cfg.SocketFile != ""
}

// UdsTokenizer communicates with a Unix Domain Socket server for tokenization.
// It implements the Tokenizer interface and manages a gRPC connection to the tokenizer service.
// The connection must be closed when the tokenizer is no longer needed by calling Close().
type UdsTokenizer struct {
	model  string
	conn   *grpc.ClientConn
	client tokenizerpb.TokenizationServiceClient
	config *UdsTokenizerConfig
}

const (
	defaultSocketFile = "/tmp/tokenizer/tokenizer-uds.socket"

	// Default timeout for requests.
	defaultTimeout = 5 * time.Second
)

// NewUdsTokenizer creates a new UDS-based tokenizer client with connection pooling.
func NewUdsTokenizer(ctx context.Context, config *UdsTokenizerConfig, modelName string) (*UdsTokenizer, error) {
	socketFile := config.SocketFile
	if socketFile == "" {
		socketFile = defaultSocketFile
	}

	// Determine address based on UseTCP flag
	var address string
	if config.UseTCP {
		// TCP address (for testing only)
		address = socketFile
	} else {
		// UDS socket path (production default)
		address = fmt.Sprintf("unix://%s", socketFile)
	}

	// Create gRPC connection
	conn, err := grpc.NewClient(
		address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{
			Time:                10 * time.Second,
			Timeout:             time.Second,
			PermitWithoutStream: true,
		}),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(100<<20), // 100MB
			grpc.MaxCallRecvMsgSize(100<<20), // 100MB
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create gRPC connection: %w", err)
	}

	client := tokenizerpb.NewTokenizationServiceClient(conn)

	udsTokenizer := &UdsTokenizer{
		conn:   conn,
		client: client,
		model:  modelName,
		config: config,
	}

	// Start a goroutine to monitor the context and close the connection when the context ends
	go func() {
		<-ctx.Done()
		udsTokenizer.Close()
	}()

	// Initialize the tokenizer for the specified model
	if err := udsTokenizer.initializeTokenizerForModel(ctx); err != nil {
		return nil, fmt.Errorf("failed to initialize tokenizer for model %s: %w", modelName, err)
	}

	return udsTokenizer, nil
}

// initializeTokenizerForModel initializes the tokenizer service for a specific model.
func (u *UdsTokenizer) initializeTokenizerForModel(ctx context.Context) error {
	config := u.config // Access the stored config

	// Use configuration values from the config - align with tokenizer_wrapper.py parameters
	req := &tokenizerpb.InitializeTokenizerRequest{
		IsLocal:     true, // Default to true per proto definition
		Model:       u.model,
		Token:       nil, // Optional - will use environment variable if needed
		DownloadDir: nil, // Optional - defaults to HF cache
	}

	if config.HuggingFaceToken != "" {
		req.Token = &config.HuggingFaceToken
	}

	if config.TokenizersCacheDir != "" {
		req.DownloadDir = &config.TokenizersCacheDir
	}

	// Retry logic with exponential backoff
	const maxRetries = 5
	const baseDelay = time.Second

	var lastErr error
	for i := 0; i < maxRetries; i++ {
		if i > 0 {
			delay := time.Duration(i) * baseDelay
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return ctx.Err()
			}
		}

		resp, err := u.client.InitializeTokenizer(ctx, req)
		if err != nil {
			lastErr = fmt.Errorf("gRPC InitializeTokenizer request failed: %w", err)
			continue
		}

		if !resp.Success {
			lastErr = fmt.Errorf("tokenizer initialization failed: %s", resp.ErrorMessage)
			continue
		}

		// Success
		return nil
	}

	return fmt.Errorf("tokenizer initialization failed after %d attempts: %w", maxRetries, lastErr)
}

// parseOffsetPairs parses the flattened array of offset pairs [start, end, start, end, ...]
// into a slice of types.Offset structs.
func parseOffsetPairs(offsetPairs []uint32) ([]types.Offset, error) {
	var tokenizersOffsets []types.Offset

	if len(offsetPairs) > 0 && len(offsetPairs)%2 == 0 {
		// Use offset_pairs field in format [start, end, start, end, ...]
		pairCount := len(offsetPairs) / 2
		tokenizersOffsets = make([]types.Offset, pairCount)
		for i := 0; i < pairCount; i++ {
			start := offsetPairs[2*i]
			end := offsetPairs[2*i+1]
			tokenizersOffsets[i] = types.Offset{uint(start), uint(end)}
		}
	} else {
		return nil, fmt.Errorf("invalid offset_pairs field in response")
	}

	return tokenizersOffsets, nil
}

func (u *UdsTokenizer) Render(prompt string) ([]uint32, []types.Offset, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	pbReq := &tokenizerpb.RenderRequest{
		Text:             prompt,
		ModelName:        u.model,
		AddSpecialTokens: true,
	}

	resp, err := u.client.Render(ctx, pbReq)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC render request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("render failed: %s", resp.ErrorMessage)
	}

	tokenizersOffsets, err := parseOffsetPairs(resp.OffsetPairs)
	if err != nil {
		return nil, nil, err
	}

	return resp.InputIds, tokenizersOffsets, nil
}

// RenderChat renders a chat template using the UDS tokenizer service.
func (u *UdsTokenizer) RenderChat(
	renderReq *types.RenderChatRequest,
) ([]uint32, []types.Offset, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	// Convert conversation messages to proto format
	messages := make([]*tokenizerpb.ChatMessage, 0, len(renderReq.Conversation))
	for _, msg := range renderReq.Conversation {
		messages = append(messages, &tokenizerpb.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	// Convert ChatTemplateKWArgs
	chatTemplateKwargs := make(map[string]*tokenizerpb.Value)
	for k, v := range renderReq.ChatTemplateKWArgs {
		chatTemplateKwargs[k] = ConvertToProtoValue(v)
	}

	// Convert tools from interface{} array to protobuf Value array
	tools := make([]*tokenizerpb.Value, 0, len(renderReq.Tools))
	for _, tool := range renderReq.Tools {
		tools = append(tools, ConvertToProtoValue(tool))
	}

	// Convert documents from interface{} array to protobuf Value array
	documents := make([]*tokenizerpb.Value, 0, len(renderReq.Documents))
	for _, doc := range renderReq.Documents {
		documents = append(documents, ConvertToProtoValue(doc))
	}

	req := &tokenizerpb.RenderChatRequest{
		Conversation:              messages,
		Tools:                     tools,
		Documents:                 documents,
		ReturnAssistantTokensMask: &renderReq.ReturnAssistantTokensMask,
		ContinueFinalMessage:      &renderReq.ContinueFinalMessage,
		AddGenerationPrompt:       &renderReq.AddGenerationPrompt,
		ChatTemplateKwargs:        chatTemplateKwargs,
		ModelName:                 u.model,
	}

	if renderReq.ChatTemplate != "" {
		req.ChatTemplate = &renderReq.ChatTemplate
	}

	resp, err := u.client.RenderChat(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC render-chat request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("render-chat failed: %s", resp.ErrorMessage)
	}

	tokenizersOffsets, err := parseOffsetPairs(resp.OffsetPairs)
	if err != nil {
		return nil, nil, err
	}

	return resp.InputIds, tokenizersOffsets, nil
}

// ConvertToProtoValue converts a Go interface{} value to a protobuf Value.
// It handles common types including strings, numbers, booleans, slices, and maps.
// Unrecognized types are converted to string representation.
func ConvertToProtoValue(v interface{}) *tokenizerpb.Value {
	if v == nil {
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_StringValue{StringValue: ""},
		}
	}

	switch val := v.(type) {
	case string:
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_StringValue{StringValue: val},
		}
	case float64:
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_NumberValue{NumberValue: val},
		}
	case bool:
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_BoolValue{BoolValue: val},
		}
	case []interface{}:
		listValues := make([]*tokenizerpb.Value, len(val))
		for i, item := range val {
			listValues[i] = ConvertToProtoValue(item)
		}
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_ListValue{ListValue: &tokenizerpb.ListValue{Values: listValues}},
		}
	case map[string]interface{}:
		structValues := make(map[string]*tokenizerpb.Value)
		for k, v := range val {
			structValues[k] = ConvertToProtoValue(v)
		}
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_StructValue{StructValue: &tokenizerpb.StructValue{Fields: structValues}},
		}
	default:
		// For unrecognized types, convert to string
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_StringValue{StringValue: fmt.Sprintf("%v", val)},
		}
	}
}

func (u *UdsTokenizer) Type() string {
	return "external-uds"
}

// Close closes the underlying gRPC connection to the tokenizer service.
func (u *UdsTokenizer) Close() error {
	if u.conn != nil {
		return u.conn.Close()
	}
	return nil
}
