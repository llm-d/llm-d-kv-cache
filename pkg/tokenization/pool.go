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
	"sync"

	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/prefixstore"
)

const (
	defaultWorkers               = 5
	defaultMinPrefixOverlapRatio = 0.8
)

// Config holds the configuration for the TokenizationPool.
type Config struct {
	// Base model name for the tokenizer.
	ModelName string `json:"modelName"`
	// Number of worker goroutines for processing tokenization tasks.
	WorkersCount int `json:"workersCount"`
	// Minimum overlap ratio to skip full tokenization and use cached prefix tokens.
	MinPrefixOverlapRatio float64 `json:"minPrefixOverlapRatio"`

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
		WorkersCount:          defaultWorkers,
		MinPrefixOverlapRatio: defaultMinPrefixOverlapRatio,
		HFTokenizerConfig:     DefaultHFTokenizerConfig(),
		LocalTokenizerConfig:  localTokenizerConfig,
	}, nil
}

// tokenizationResponse holds the result of a tokenization operation.
type tokenizationResponse struct {
	Tokens []uint32
}

// Task represents a unit of work for tokenizing a prompt.
type Task struct {
	RenderReq *preprocessing.ApplyChatTemplateRequest
	Prompt    string
	ModelName string
	ResultCh  chan<- tokenizationResponse // nil => fire-and-forget
}

// Pool encapsulates the queue, worker pool, and token indexer.
type Pool struct {
	modelName string // base model name for tokenization
	workers   int
	queue     workqueue.TypedRateLimitingInterface[Task]
	wg        sync.WaitGroup
	indexer   prefixstore.Indexer

	// Tokenizer is configured for the specific model this pool handles.
	// It's shared between all pool workers. Since the tokenizer
	// is immutable, Encode calls are safe for concurrent use without locks.
	tokenizer Tokenizer

	// Minimum overlap ratio to skip full tokenization and use cached prefix tokens.
	minPrefixOverlapRatio float64
}

// NewTokenizationPool initializes a TokenizationPool with the specified number
// of workers and the provided Indexer.
func NewTokenizationPool(ctx context.Context, config *Config, store prefixstore.Indexer) (*Pool, error) {
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
		modelName:             config.ModelName,
		workers:               config.WorkersCount,
		queue:                 workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[Task]()),
		indexer:               store,
		tokenizer:             &CompositeTokenizer{Tokenizers: tokenizers},
		minPrefixOverlapRatio: config.MinPrefixOverlapRatio,
	}, nil
}

// EnqueueTokenization enqueues a new tokenization task.
// This method only enqueues the task and does not start processing it.
func (pool *Pool) EnqueueTokenization(prompt string) {
	task := Task{
		Prompt: prompt,
	}
	pool.queue.Add(task)
}

// Tokenize queues a task and blocks until the final result is available.
func (pool *Pool) Tokenize(renderReq *preprocessing.ApplyChatTemplateRequest, prompt string) []uint32 {
	resultCh := make(chan tokenizationResponse, 1)
	pool.queue.Add(Task{
		RenderReq: renderReq,
		Prompt:    prompt,
		ResultCh:  resultCh,
	})

	res := <-resultCh
	tokens := res.Tokens
	return tokens
}

// Run launches worker goroutines that process tasks until the context is
// cancelled.
func (pool *Pool) Run(ctx context.Context) {
	for i := 0; i < pool.workers; i++ {
		pool.wg.Add(1)
		go pool.workerLoop(i)
	}

	<-ctx.Done()

	pool.queue.ShutDown()
	pool.wg.Wait()
}

// workerLoop is the main processing loop for each worker.
func (pool *Pool) workerLoop(_ int) {
	defer pool.wg.Done()
	// max number of times to retry a failed task before dropping it.
	const maxRetries = 3

	for {
		task, shutdown := pool.queue.Get()
		if shutdown {
			return
		}

		// Process the task.
		err := pool.processTask(task)
		switch {
		case err == nil:
			pool.queue.Forget(task)
		case pool.queue.NumRequeues(task) < maxRetries:
			pool.queue.AddRateLimited(task)
		default:
			// Retries exceeded. Drop the task and unblock the caller.
			log.Log.Error(err, "Dropping tokenization task after max retries", "prompt", task.Prompt,
				"retries", maxRetries)
			pool.queue.Forget(task)
			if task.ResultCh != nil {
				// Closing the channel signals failure (zero value received by caller)
				close(task.ResultCh)
			}
		}
		pool.queue.Done(task)
	}
}

// processTask tokenizes the prompt and updates the indexer.
// It sends exactly one response (success or error) if ResultCh is provided.
func (pool *Pool) processTask(task Task) error {
	// https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/entrypoints/openai/protocol.py#L1127
	addSpecialTokens := true
	if task.RenderReq != nil {
		var err error
		task.Prompt, err = pool.tokenizer.ApplyChatTemplate(pool.modelName, task.RenderReq)
		if err != nil {
			log.Log.Error(err, "failed to render chat template")
			return err
		}
		// https://github.com/vllm-project/vllm/blob/v0.11.2/vllm/entrypoints/openai/protocol.py#L613
		addSpecialTokens = false
	}

	tokenIDs, overlapRatio := pool.indexer.FindLongestContainedTokens(task.Prompt)

	// if the overlap ratio is low, get the full tokenization
	if overlapRatio < pool.minPrefixOverlapRatio {
		tokens, offsets, err := pool.tokenizer.Encode(pool.modelName, &preprocessing.EncodeRequest{
			Text:             task.Prompt,
			AddSpecialTokens: addSpecialTokens,
		})
		if err != nil {
			log.Log.Error(err, "failed to encode tokens", "prompt", task.Prompt)
			return err
		}

		// update the indexer with the new tokenization
		if e := pool.indexer.AddTokenization(task.Prompt, tokens, offsets); e != nil {
			err = fmt.Errorf("tokenization failed for model %s: %w", task.ModelName, e)
			return err
		}

		tokenIDs = tokens
	}

	// On success, send the response if a channel is provided and close the channel.
	if task.ResultCh != nil {
		resp := tokenizationResponse{
			Tokens: tokenIDs,
		}
		task.ResultCh <- resp
		close(task.ResultCh)
	}

	return nil
}

func (pool *Pool) SetTokenizer(tokenizer Tokenizer, modelName string) {
	pool.tokenizer = tokenizer
	pool.modelName = modelName
}
