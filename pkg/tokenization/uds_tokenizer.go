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
	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
)

// UdsTokenizerConfig represents the configuration for the UDS-based tokenizer,
// including the socket file path.
type UdsTokenizerConfig struct {
	SocketFile string `json:"socketFile"`
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
}

const (
	defaultSocketFile = "/tmp/tokenizer/tokenizer-uds.socket"

	// Default timeout for requests.
	defaultTimeout = 5 * time.Second
)

// NewUdsTokenizer creates a new UDS-based tokenizer client with connection pooling.
func NewUdsTokenizer(ctx context.Context, config *UdsTokenizerConfig, modelName string) (Tokenizer, error) {
	socketFile := config.SocketFile
	if socketFile == "" {
		socketFile = defaultSocketFile
	}

	// Create gRPC connection using UDS
	conn, err := grpc.NewClient(
		fmt.Sprintf("unix://%s", socketFile),
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
	// Use default configuration values for now
	req := &tokenizerpb.InitializeTokenizerRequest{
		ModelName:           u.model,
		EnableThinking:      false, // Can be made configurable later
		AddGenerationPrompt: true,  // Can be made configurable later
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

func (u *UdsTokenizer) Render(prompt string) ([]uint32, []preprocessing.Offset, error) {
	return u.Encode(prompt, true)
}

// Encode tokenizes the input string and returns the token IDs and offsets.
func (u *UdsTokenizer) Encode(prompt string, addSpecialTokens bool) ([]uint32, []preprocessing.Offset, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	pbReq := &tokenizerpb.TokenizeRequest{
		Input:            prompt,
		ModelName:        u.model,
		AddSpecialTokens: addSpecialTokens,
	}

	resp, err := u.client.Tokenize(ctx, pbReq)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC tokenize request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("tokenization failed: %s", resp.ErrorMessage)
	}

	// Use offset_pairs field in format [start, end, start, end, ...]
	var tokenizersOffsets []preprocessing.Offset

	if len(resp.OffsetPairs) > 0 && len(resp.OffsetPairs)%2 == 0 {
		// Use offset_pairs field in format [start, end, start, end, ...]
		pairCount := len(resp.OffsetPairs) / 2
		tokenizersOffsets = make([]preprocessing.Offset, pairCount)
		for i := 0; i < pairCount; i++ {
			start := resp.OffsetPairs[2*i]
			end := resp.OffsetPairs[2*i+1]
			tokenizersOffsets[i] = preprocessing.Offset{uint(start), uint(end)}
		}
	} else {
		return nil, nil, fmt.Errorf("invalid offset_pairs field in response")
	}

	return resp.InputIds, tokenizersOffsets, nil
}

// RenderChat renders a chat template using the UDS tokenizer service.
func (u *UdsTokenizer) RenderChat(
	renderReq *preprocessing.RenderChatRequest,
) ([]uint32, []preprocessing.Offset, error) {
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
	conversationTurns := []*tokenizerpb.ConversationTurn{
		{Messages: messages},
	}

	// Convert ChatTemplateKWArgs
	chatTemplateKwargs := make(map[string]*tokenizerpb.Value)
	for k, v := range renderReq.ChatTemplateKWArgs {
		chatTemplateKwargs[k] = convertToProtoValue(v)
	}

	req := &tokenizerpb.ChatTemplateRequest{
		ConversationTurns:         conversationTurns,
		ChatTemplate:              renderReq.ChatTemplate,
		ReturnAssistantTokensMask: renderReq.ReturnAssistantTokensMask,
		ContinueFinalMessage:      renderReq.ContinueFinalMessage,
		AddGenerationPrompt:       renderReq.AddGenerationPrompt,
		ChatTemplateKwargs:        chatTemplateKwargs,
		ModelName:                 u.model,
	}

	resp, err := u.client.RenderChatTemplate(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC chat-template request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("chat template rendering failed: %s", resp.ErrorMessage)
	}

	return u.Encode(resp.RenderedPrompt, false)
}

func convertToProtoValue(v interface{}) *tokenizerpb.Value {
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
			listValues[i] = convertToProtoValue(item)
		}
		return &tokenizerpb.Value{
			Value: &tokenizerpb.Value_ListValue{ListValue: &tokenizerpb.ListValue{Values: listValues}},
		}
	case map[string]interface{}:
		structValues := make(map[string]*tokenizerpb.Value)
		for k, v := range val {
			structValues[k] = convertToProtoValue(v)
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
