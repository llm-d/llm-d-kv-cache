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

	"github.com/daulet/tokenizers"
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
type UdsTokenizer struct {
	conn   *grpc.ClientConn
	client tokenizerpb.TokenizationServiceClient
}

const (
	defaultSocketFile = "/tmp/tokenizer/tokenizer-uds.socket"

	// Default timeout for requests.
	defaultTimeout = 5 * time.Second
)

// NewUdsTokenizer creates a new UDS-based tokenizer client with connection pooling.
func NewUdsTokenizer(_ context.Context, config *UdsTokenizerConfig) (Tokenizer, error) {
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

	return &UdsTokenizer{
		conn:   conn,
		client: client,
	}, nil
}

// Encode tokenizes the input string and returns the token IDs and offsets.
func (u *UdsTokenizer) Encode(input, modelName string, _ bool) ([]uint32, []tokenizers.Offset, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	req := &tokenizerpb.TokenizeRequest{
		Input:     input,
		ModelName: modelName,
	}

	resp, err := u.client.Tokenize(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("gRPC tokenize request failed: %w", err)
	}

	if !resp.Success {
		return nil, nil, fmt.Errorf("tokenization failed: %s", resp.ErrorMessage)
	}

	// Use offset_pairs field in format [start, end, start, end, ...]
	var tokenizersOffsets []tokenizers.Offset

	if len(resp.OffsetPairs) > 0 && len(resp.OffsetPairs)%2 == 0 {
		// Use offset_pairs field in format [start, end, start, end, ...]
		pairCount := len(resp.OffsetPairs) / 2
		tokenizersOffsets = make([]tokenizers.Offset, pairCount)
		for i := 0; i < pairCount; i++ {
			start := resp.OffsetPairs[2*i]
			end := resp.OffsetPairs[2*i+1]
			tokenizersOffsets[i] = tokenizers.Offset{uint(start), uint(end)}
		}
	}

	return resp.InputIds, tokenizersOffsets, nil
}

// ApplyChatTemplate renders a chat template using the UDS tokenizer service.
func (u *UdsTokenizer) ApplyChatTemplate(
	modelName string, renderReq *preprocessing.ApplyChatTemplateRequest,
) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	// Convert preprocessing.ChatMessage to protobuf ChatMessage
	chatMessages := make([]*tokenizerpb.ChatMessage, len(renderReq.Conversations))
	for i, msg := range renderReq.Conversations {
		chatMessages[i] = &tokenizerpb.ChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Convert ChatTemplateKWArgs
	chatTemplateKwargs := make(map[string]*tokenizerpb.Value)
	for k, v := range renderReq.ChatTemplateKWArgs {
		chatTemplateKwargs[k] = convertToProtoValue(v)
	}

	req := &tokenizerpb.ChatTemplateRequest{
		Messages:                  chatMessages,
		ChatTemplate:              renderReq.ChatTemplate,
		ReturnAssistantTokensMask: renderReq.ReturnAssistantTokensMask,
		ContinueFinalMessage:      renderReq.ContinueFinalMessage,
		AddGenerationPrompt:       renderReq.AddGenerationPrompt,
		ChatTemplateKwargs:        chatTemplateKwargs,
		ModelName:                 modelName,
	}

	resp, err := u.client.RenderChatTemplate(ctx, req)
	if err != nil {
		return "", fmt.Errorf("gRPC chat-template request failed: %w", err)
	}

	if !resp.Success {
		return "", fmt.Errorf("chat template rendering failed: %s", resp.ErrorMessage)
	}

	return resp.RenderedPrompt, nil
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

func (u *UdsTokenizer) Close() error {
	if u.conn != nil {
		return u.conn.Close()
	}
	return nil
}
