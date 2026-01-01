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
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"strings"
	"time"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"golang.org/x/net/http2"
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
	httpClient *http.Client
	baseURL    string
}

// TokenizedInput represents the response from the tokenize endpoint.
type TokenizedInput struct {
	InputIDs      []uint32               `json:"input_ids"`
	OffsetMapping []preprocessing.Offset `json:"offset_mapping"`
}

const (
	defaultSocketFile = "/tmp/tokenizer/tokenizer-uds.socket"
	baseURL           = "http://tokenizer"

	// Default timeout for requests.
	defaultTimeout    = 5 * time.Second
	defaultMaxRetries = 2

	// Initial delay for exponential backoff.
	initialRetryDelay = 100 * time.Millisecond
)

// NewUdsTokenizer creates a new UDS-based tokenizer client with connection pooling.
func NewUdsTokenizer(config *UdsTokenizerConfig) (Tokenizer, error) {
	dialer := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}

	socketFile := config.SocketFile
	if socketFile == "" {
		socketFile = defaultSocketFile
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return dialer.DialContext(ctx, "unix", socketFile)
		},
		MaxIdleConns:          100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	if err := http2.ConfigureTransport(transport); err != nil {
		return nil, fmt.Errorf("failed to configure HTTP/2 transport: %w", err)
	}

	client := &http.Client{
		Transport: transport,
	}

	return &UdsTokenizer{
		httpClient: client,
		baseURL:    baseURL,
	}, nil
}

// Encode tokenizes the input string and returns the token IDs and offsets.
func (u *UdsTokenizer) Encode(req *preprocessing.EncodeRequest) ([]uint32, []preprocessing.Offset, error) {
	httpReq, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		u.baseURL+"/tokenize",
		strings.NewReader(req.Text),
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to create request: %w", err)
	}

	respBody, err := u.executeRequest(httpReq, defaultTimeout, defaultMaxRetries)
	if err != nil {
		return nil, nil, fmt.Errorf("tokenize request failed: %w", err)
	}

	var tokenized TokenizedInput
	if err := json.Unmarshal(respBody, &tokenized); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	return tokenized.InputIDs, tokenized.OffsetMapping, nil
}

// ApplyChatTemplate renders a chat template using the UDS tokenizer service.
func (u *UdsTokenizer) ApplyChatTemplate(
	_ string, renderReq *preprocessing.ApplyChatTemplateRequest,
) (string, error) {
	messagesBytes, err := json.Marshal(renderReq.Conversation)
	if err != nil {
		return "", fmt.Errorf("failed to marshal chat-completions messages: %w", err)
	}

	req, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		u.baseURL+"/chat-template",
		bytes.NewBuffer(messagesBytes),
	)
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	respBody, err := u.executeRequest(req, defaultTimeout, defaultMaxRetries)
	if err != nil {
		return "", fmt.Errorf("chat-template request failed: %w", err)
	}
	return string(respBody), nil
}

func (u *UdsTokenizer) Type() string {
	return "external-uds"
}

// executeRequest executes an HTTP request with timeout and retry logic.
func (u *UdsTokenizer) executeRequest(
	req *http.Request,
	timeout time.Duration,
	maxRetries int,
) ([]byte, error) {
	if timeout == 0 {
		timeout = defaultTimeout
	}
	if maxRetries < 0 {
		maxRetries = defaultMaxRetries
	}

	// Try the request up to maxRetries+1 times
	delay := initialRetryDelay
	var lastErr error

	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Create a context with timeout
		ctx, cancel := context.WithTimeout(req.Context(), timeout)
		reqWithCtx := req.WithContext(ctx)

		// Execute the request
		resp, err := u.httpClient.Do(reqWithCtx)

		// Process the response if no error occurred
		if err == nil {
			body, processErr := u.processResponse(resp, cancel)
			if processErr == nil {
				return body, nil
			}
			lastErr = processErr
		} else {
			lastErr = err
			cancel()
		}

		// If this was the last attempt, break
		if attempt == maxRetries {
			break
		}

		// Wait before retrying with exponential backoff
		time.Sleep(delay)
		delay *= 2 // Exponential backoff

		// Add some jitter to prevent thundering herd
		jitter, randErr := rand.Int(rand.Reader, big.NewInt(int64(delay/2)))
		if randErr != nil {
			// Fallback to using the full delay without jitter
			jitter = big.NewInt(int64(delay / 2))
		}
		delay += time.Duration(jitter.Int64())
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("request failed after %d attempts", maxRetries+1)
	} else {
		lastErr = fmt.Errorf("request failed after %d attempts: %w", maxRetries+1, lastErr)
	}
	return nil, lastErr
}

// processResponse handles the HTTP response, checking status code and reading the body.
func (u *UdsTokenizer) processResponse(resp *http.Response, cancel context.CancelFunc) ([]byte, error) {
	defer cancel()

	// For 200 status codes, don't retry.
	if resp.StatusCode == http.StatusOK {
		// Read the response before canceling the context
		body, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr == nil {
			return body, nil
		}
		return nil, fmt.Errorf("failed to read response body: %w", readErr)
	}

	// For non-200 status codes, close the response body and return an error
	errorMsg := fmt.Errorf("server returned status %d", resp.StatusCode)
	resp.Body.Close()
	return nil, errorMsg
}
