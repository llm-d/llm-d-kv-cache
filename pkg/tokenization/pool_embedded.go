//go:build embedded_tokenizers

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

	"k8s.io/client-go/util/workqueue"
)

// Config holds the configuration for the TokenizationPool.
type Config struct {
	// Base model name for the tokenizer.
	ModelName string `json:"modelName"`
	// Number of worker goroutines for processing tokenization tasks.
	WorkersCount int `json:"workersCount"`

	LocalTokenizerConfig *LocalTokenizerConfig `json:"local,omitempty"`
	UdsTokenizerConfig   *UdsTokenizerConfig   `json:"uds,omitempty"`
	HFTokenizerConfig    *HFTokenizerConfig    `json:"hf,omitempty"`
}

// DefaultConfig returns a default configuration for the TokenizationPool.
func DefaultConfig() (*Config, error) {
	localTokenizerConfig, err := DefaultLocalTokenizerConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create default local tokenizer config: %w", err)
	}

	return &Config{
		WorkersCount:         defaultWorkers,
		HFTokenizerConfig:    DefaultHFTokenizerConfig(),
		LocalTokenizerConfig: localTokenizerConfig,
	}, nil
}

// NewTokenizationPool initializes a TokenizationPool with the specified number
// of workers and the provided configuration.
func NewTokenizationPool(ctx context.Context, config *Config) (*Pool, error) {
	if config == nil || config.ModelName == "" {
		return nil, fmt.Errorf("config and config.ModelName cannot be nil or empty")
	}

	if !config.LocalTokenizerConfig.IsEnabled() &&
		!config.UdsTokenizerConfig.IsEnabled() &&
		!config.HFTokenizerConfig.IsEnabled() {
		return nil, fmt.Errorf("at least one tokenizer config must be enabled")
	}

	tokenizers := make([]Tokenizer, 0, 3)

	if config.LocalTokenizerConfig.IsEnabled() {
		localTokenizer, err := NewCachedLocalTokenizer(ctx,
			config.ModelName, *config.LocalTokenizerConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create local tokenizer: %w", err)
		}
		tokenizers = append(tokenizers, localTokenizer)
	}

	if config.UdsTokenizerConfig.IsEnabled() {
		udsTokenizer, err := NewUdsTokenizer(ctx,
			config.UdsTokenizerConfig, config.ModelName)
		if err != nil {
			return nil, fmt.Errorf("failed to create UDS tokenizer: %w", err)
		}
		tokenizers = append(tokenizers, udsTokenizer)
	}

	if config.HFTokenizerConfig.IsEnabled() {
		hfTokenizer, err := NewCachedHFTokenizer(ctx,
			config.ModelName, config.HFTokenizerConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create HuggingFace tokenizer: %w", err)
		}
		tokenizers = append(tokenizers, hfTokenizer)
	}

	return &Pool{
		modelName: config.ModelName,
		workers:   config.WorkersCount,
		queue:     workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[Task]()),
		tokenizer: &CompositeTokenizer{Tokenizers: tokenizers},
	}, nil
}
