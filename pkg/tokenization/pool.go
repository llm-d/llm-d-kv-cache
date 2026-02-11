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
	"sync"

	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

const (
	defaultWorkers = 5
)

// tokenizationResponse holds the result of a tokenization operation.
type tokenizationResponse struct {
	Tokens []uint32
}

// Task represents a unit of work for tokenizing a prompt.
type Task struct {
	RenderReq *types.RenderChatRequest
	Prompt    string
	ModelName string
	ResultCh  chan<- tokenizationResponse // nil => fire-and-forget
}

// Pool encapsulates the queue and worker pool for tokenization tasks.
type Pool struct {
	modelName string // base model name for tokenization
	workers   int
	queue     workqueue.TypedRateLimitingInterface[Task]
	wg        sync.WaitGroup

	// Tokenizer is configured for the specific model this pool handles.
	// It's shared between all pool workers.
	// Since the tokenizer is immutable,
	// tokenizer functions are safe for concurrent use without locks.
	tokenizer Tokenizer
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
func (pool *Pool) Tokenize(renderReq *types.RenderChatRequest, prompt string) []uint32 {
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

// processTask tokenizes the prompt and returns the tokens via ResultCh.
// It sends exactly one response (success or error) if ResultCh is provided.
func (pool *Pool) processTask(task Task) error {
	var tokens []uint32
	var err error
	if task.RenderReq == nil {
		tokens, _, err = pool.tokenizer.Render(task.Prompt)
		if err != nil {
			log.Log.Error(err, "failed to render tokens", "prompt", task.Prompt)
			return err
		}
	} else {
		tokens, _, err = pool.tokenizer.RenderChat(task.RenderReq)
		if err != nil {
			log.Log.Error(err, "failed to render tokens", "task", task.RenderReq)
			return err
		}
	}

	// On success, send the response if a channel is provided and close the channel.
	if task.ResultCh != nil {
		resp := tokenizationResponse{
			Tokens: tokens,
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
