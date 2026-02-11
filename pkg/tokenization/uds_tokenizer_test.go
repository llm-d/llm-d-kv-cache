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

package tokenization_test

import (
	"context"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"testing"

	tokenizerpb "github.com/llm-d/llm-d-kv-cache/api/tokenizerpb"
	. "github.com/llm-d/llm-d-kv-cache/pkg/tokenization"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
	"google.golang.org/grpc"
)

// mockTokenizationServer implements the TokenizationServiceServer interface for testing.
type mockTokenizationServer struct {
	tokenizerpb.UnimplementedTokenizationServiceServer
	initializeError bool
	tokenizeError   bool
	chatError       bool
	initialized     map[string]bool
}

func newMockTokenizationServer() *mockTokenizationServer {
	return &mockTokenizationServer{
		initialized: make(map[string]bool),
	}
}

func (m *mockTokenizationServer) InitializeTokenizer(
	ctx context.Context,
	req *tokenizerpb.InitializeTokenizerRequest,
) (*tokenizerpb.InitializeTokenizerResponse, error) {
	if m.initializeError {
		return &tokenizerpb.InitializeTokenizerResponse{
			Success:      false,
			ErrorMessage: "mock initialization error",
		}, nil
	}

	m.initialized[req.ModelName] = true
	return &tokenizerpb.InitializeTokenizerResponse{
		Success: true,
	}, nil
}

func (m *mockTokenizationServer) Tokenize(
	ctx context.Context,
	req *tokenizerpb.TokenizeRequest,
) (*tokenizerpb.TokenizeResponse, error) {
	if m.tokenizeError {
		return &tokenizerpb.TokenizeResponse{
			Success:      false,
			ErrorMessage: "mock tokenization error",
		}, nil
	}

	// Check if model was initialized (matches real service behavior)
	if !m.initialized[req.ModelName] {
		return &tokenizerpb.TokenizeResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("model %s not initialized", req.ModelName),
		}, nil
	}

	// Simple deterministic mock tokenization: convert each rune to a token ID
	// This makes tests more realistic - different inputs produce different tokens
	input := req.Input
	tokens := make([]uint32, 0, len(input))
	offsets := make([]uint32, 0, len(input)*2)

	for i, r := range input {
		tokens = append(tokens, uint32(r))
		// #nosec G115 -- i is bounded by string length, safe conversion
		offsets = append(offsets, uint32(i), uint32(i+1))
	}

	return &tokenizerpb.TokenizeResponse{
		InputIds:     tokens,
		Success:      true,
		OffsetPairs:  offsets,
		ErrorMessage: "",
	}, nil
}

func (m *mockTokenizationServer) RenderChatTemplate(
	ctx context.Context,
	req *tokenizerpb.ChatTemplateRequest,
) (*tokenizerpb.ChatTemplateResponse, error) {
	if m.chatError {
		return &tokenizerpb.ChatTemplateResponse{
			Success:      false,
			ErrorMessage: "mock chat template error",
		}, nil
	}

	// Check if model was initialized (matches real service behavior)
	if !m.initialized[req.ModelName] {
		return &tokenizerpb.ChatTemplateResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("model %s not initialized", req.ModelName),
		}, nil
	}

	// Mock chat template rendering by concatenating messages
	rendered := ""
	for _, turn := range req.ConversationTurns {
		for _, msg := range turn.Messages {
			rendered += fmt.Sprintf("%s: %s\n", msg.Role, msg.Content)
		}
	}

	return &tokenizerpb.ChatTemplateResponse{
		RenderedPrompt: rendered,
		Success:        true,
	}, nil
}

// UdsTokenizerTestSuite holds the test suite state.
type UdsTokenizerTestSuite struct {
	suite.Suite
	mockServer *mockTokenizationServer
	socketPath string
	grpcServer *grpc.Server
	listener   net.Listener
	tokenizer  *UdsTokenizer // Shared tokenizer for most tests
}

// SetupSuite runs once before all tests in the suite.
func (s *UdsTokenizerTestSuite) SetupSuite() {
	s.mockServer = newMockTokenizationServer()

	tmpDir, err := os.MkdirTemp("", "tok-test-")
	require.NoError(s.T(), err)
	s.socketPath = filepath.Join(tmpDir, "tokenizer-uds.sock")

	// Create a Unix listener
	lc := net.ListenConfig{}
	listener, err := lc.Listen(context.Background(), "unix", s.socketPath)
	require.NoError(s.T(), err)
	s.listener = listener

	// Create and start the gRPC server
	s.grpcServer = grpc.NewServer()
	tokenizerpb.RegisterTokenizationServiceServer(s.grpcServer, s.mockServer)

	go func() {
		if err := s.grpcServer.Serve(s.listener); err != nil {
			s.T().Logf("Server error: %v", err)
		}
	}()

	config := &UdsTokenizerConfig{SocketFile: s.socketPath}
	s.tokenizer, err = NewUdsTokenizer(s.T().Context(), config, "test-model")
	require.NoError(s.T(), err)
	require.NotNil(s.T(), s.tokenizer)
}

// TearDownSuite runs once after all tests in the suite.
func (s *UdsTokenizerTestSuite) TearDownSuite() {
	// Close client connection first for graceful shutdown
	if s.tokenizer != nil {
		s.tokenizer.Close()
	}

	// Then stop the server
	if s.grpcServer != nil {
		s.grpcServer.Stop()
	}
	if s.listener != nil {
		s.listener.Close()
	}

	// Clean up the temp directory
	if s.socketPath != "" {
		os.RemoveAll(filepath.Dir(s.socketPath))
	}
}

// SetupTest runs before each test to reset mock state.
func (s *UdsTokenizerTestSuite) SetupTest() {
	// Reset error flags for each test
	s.mockServer.initializeError = false
	s.mockServer.tokenizeError = false
	s.mockServer.chatError = false
	// Clear initialized models to ensure test isolation
	s.mockServer.initialized = make(map[string]bool)
	// Re-initialize the shared tokenizer's model
	if s.tokenizer != nil {
		s.mockServer.initialized["test-model"] = true
	}
}

func TestUdsTokenizerSuite(t *testing.T) {
	suite.Run(t, new(UdsTokenizerTestSuite))
}

func (s *UdsTokenizerTestSuite) TestNewUdsTokenizer_InitializationFailure() {
	s.mockServer.initializeError = true

	config := &UdsTokenizerConfig{
		SocketFile: s.socketPath,
	}

	tokenizer, err := NewUdsTokenizer(s.T().Context(), config, "test-model")
	s.Assert().Error(err)
	s.Assert().Nil(tokenizer)
}

func (s *UdsTokenizerTestSuite) TestNewUdsTokenizer_ConnectionFailure() {
	config := &UdsTokenizerConfig{
		SocketFile: "/non/existent/socket.sock",
	}

	// This should fail quickly because the socket doesn't exist
	tokenizer, err := NewUdsTokenizer(s.T().Context(), config, "test-model")
	s.Assert().Error(err)
	s.Assert().Nil(tokenizer)
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_Render() {
	// Test Render - character-based tokenization
	input := "hello world"
	tokens, offsets, err := s.tokenizer.Render(input)
	s.Require().NoError(err)

	// Each character becomes a token
	s.Assert().Equal(len([]rune(input)), len(tokens))
	s.Assert().Equal(len([]rune(input)), len(offsets))

	// Verify specific characters
	s.Assert().Equal(uint32('h'), tokens[0])  // 'h' = 104
	s.Assert().Equal(uint32(' '), tokens[5])  // space at position 5 = 32
	s.Assert().Equal(uint32('d'), tokens[10]) // 'd' at end = 100

	// Verify offsets
	s.Assert().Equal(types.Offset{0, 1}, offsets[0]) // 'h'
	s.Assert().Equal(types.Offset{5, 6}, offsets[5]) // space
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_RenderChat() {
	// Test RenderChat
	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hi there"},
		},
		AddGenerationPrompt: true,
		ChatTemplateKWArgs: map[string]interface{}{
			"key1": "value1",
			"key2": float64(42),
		},
	}

	tokens, offsets, err := s.tokenizer.RenderChat(renderReq)
	s.Require().NoError(err)
	s.Assert().Greater(len(tokens), 0, "should return tokens from rendered chat")
	s.Assert().Equal(len(tokens), len(offsets), "offsets should match token count")
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_Type() {
	s.Assert().Equal("external-uds", s.tokenizer.Type())
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_TokenizeError() {
	s.mockServer.tokenizeError = true

	_, _, err := s.tokenizer.Render("test")
	s.Assert().Error(err)
	s.Assert().Contains(err.Error(), "tokenization failed")
}

func (s *UdsTokenizerTestSuite) TestUdsTokenizer_ChatTemplateError() {
	s.mockServer.chatError = true

	renderReq := &types.RenderChatRequest{
		Conversation: []types.Conversation{
			{Role: "user", Content: "Hello"},
		},
	}

	_, _, err := s.tokenizer.RenderChat(renderReq)
	s.Assert().Error(err)
	s.Assert().Contains(err.Error(), "chat template rendering failed")
}

// convertFromProtoValue converts a proto Value back to a Go interface{} value.
// This is used for testing round-trip conversions.
func convertFromProtoValue(pv *tokenizerpb.Value) interface{} {
	if pv == nil {
		return nil
	}

	switch v := pv.Value.(type) {
	case *tokenizerpb.Value_StringValue:
		return v.StringValue
	case *tokenizerpb.Value_NumberValue:
		return v.NumberValue
	case *tokenizerpb.Value_BoolValue:
		return v.BoolValue
	case *tokenizerpb.Value_ListValue:
		result := make([]interface{}, len(v.ListValue.Values))
		for i, item := range v.ListValue.Values {
			result[i] = convertFromProtoValue(item)
		}
		return result
	case *tokenizerpb.Value_StructValue:
		result := make(map[string]interface{})
		for k, val := range v.StructValue.Fields {
			result[k] = convertFromProtoValue(val)
		}
		return result
	default:
		return nil
	}
}

func TestConvertToProtoValue(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected interface{} // Only set when different from input
	}{
		{
			name:     "nil value converts to empty string",
			input:    nil,
			expected: "",
		},
		{
			name:  "string value",
			input: "test string",
		},
		{
			name:  "empty string",
			input: "",
		},
		{
			name:  "float64 value",
			input: 42.5,
		},
		{
			name:  "zero float64",
			input: float64(0),
		},
		{
			name:  "negative float64",
			input: -123.456,
		},
		{
			name:  "bool true",
			input: true,
		},
		{
			name:  "bool false",
			input: false,
		},
		{
			name:  "empty slice",
			input: []interface{}{},
		},
		{
			name:  "simple slice",
			input: []interface{}{"a", "b", "c"},
		},
		{
			name:  "mixed slice",
			input: []interface{}{"string", 42.0, true, false},
		},
		{
			name:  "nested slice",
			input: []interface{}{[]interface{}{"nested", 1.0}, []interface{}{2.0, "values"}},
		},
		{
			name:  "empty map",
			input: map[string]interface{}{},
		},
		{
			name:  "simple map",
			input: map[string]interface{}{"key1": "value1", "key2": "value2"},
		},
		{
			name: "mixed map",
			input: map[string]interface{}{
				"string": "text",
				"number": 42.0,
				"bool":   true,
			},
		},
		{
			name: "nested map",
			input: map[string]interface{}{
				"outer": map[string]interface{}{
					"inner1": "value1",
					"inner2": 123.0,
				},
			},
		},
		{
			name: "complex nested structure",
			input: map[string]interface{}{
				"users": []interface{}{
					map[string]interface{}{"name": "Alice", "age": 30.0},
					map[string]interface{}{"name": "Bob", "age": 25.0},
				},
				"metadata": map[string]interface{}{
					"version": "1.0",
					"active":  true,
					"tags":    []interface{}{"tag1", "tag2"},
				},
			},
		},
		{
			name:     "unrecognized type int converts to string",
			input:    int(42),
			expected: "42",
		},
		{
			name:     "unrecognized type struct converts to string",
			input:    struct{ name string }{name: "test"},
			expected: "{test}",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Convert to proto
			protoValue := ConvertToProtoValue(tt.input)
			require.NotNil(t, protoValue)

			// Convert back to Go value
			result := convertFromProtoValue(protoValue)

			// Determine expected value
			expected := tt.expected
			if expected == nil {
				expected = tt.input
			}

			// Verify round-trip conversion
			assert.Equal(t, expected, result)
		})
	}
}
