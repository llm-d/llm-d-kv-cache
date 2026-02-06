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
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"go.uber.org/multierr"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/metrics"
	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
)

// Tokenizer interface defines the methods for tokenization.
type Tokenizer interface {
	RenderChat(*preprocessing.RenderChatRequest) ([]uint32, []preprocessing.Offset, error)
	Render(string) ([]uint32, []preprocessing.Offset, error)
	Type() string
}

// HFTokenizerConfig holds the configuration for the HuggingFace tokenizer.
type HFTokenizerConfig struct {
	Enabled            bool   `json:"enabled"`
	HuggingFaceToken   string `json:"huggingFaceToken"`
	TokenizersCacheDir string `json:"tokenizersCacheDir"` // Directory for caching tokenizers
}

func (cfg *HFTokenizerConfig) IsEnabled() bool {
	return cfg != nil && cfg.Enabled
}

// DefaultHFTokenizerConfig returns a default configuration for the HuggingFace
// tokenizer.
func DefaultHFTokenizerConfig() *HFTokenizerConfig {
	return &HFTokenizerConfig{
		Enabled:            true,
		HuggingFaceToken:   "",
		TokenizersCacheDir: getTokenizerCacheDir(),
	}
}

//nolint:gosec // These are environment variable names, not credentials
const (
	localTokenizerDirEnv      = "LOCAL_TOKENIZER_DIR"
	localTokenizerFileNameEnv = "LOCAL_TOKENIZER_FILENAME"
)

// localTokenizerDir is the base directory for local tokenizer files.
// It can be set via the LOCAL_TOKENIZER_DIR environment variable.
// If not set, it defaults to defaultLocalTokenizerDir.
var (
	localTokenizerDir      = os.Getenv(localTokenizerDirEnv)
	localTokenizerFileName = os.Getenv(localTokenizerFileNameEnv)
)

// defaultLocalTokenizerDir is the default directory to search for local tokenizer files.
// This is typically used in containerized environments where models are mounted at /mnt/models.
//
//nolint:gosec // These are default paths, not credentials
const (
	defaultLocalTokenizerDir      = "/mnt/models"
	defaultLocalTokenizerFileName = "tokenizer.json"
)

func init() {
	if localTokenizerDir == "" {
		localTokenizerDir = defaultLocalTokenizerDir
	}
	if localTokenizerFileName == "" {
		localTokenizerFileName = defaultLocalTokenizerFileName
	}
}

// LocalTokenizerConfig provides a mapping from model names to local tokenizer.json file paths.
// This allows the system to use pre-downloaded tokenizer files instead of fetching them from HuggingFace,
// which is useful for air-gapped environments or when models are preloaded on disk.
type LocalTokenizerConfig struct {
	// AutoDiscoveryDir is the local directory to search for local tokenizer files.
	// Default: defaultLocalTokenizerDir.
	AutoDiscoveryDir string `json:"autoDiscoveryDir,omitempty"`
	// AutoDiscoveryTokenizerFile is the local file to search for local tokenizer files.
	// Default: defaultLocalTokenizerFileName
	AutoDiscoveryTokenizerFileName string `json:"autoDiscoveryTokenizerFileName,omitempty"`

	// ModelTokenizerMap is a map from model name to the absolute path of its tokenizer.json file.
	// The model name (key) is typically the directory name containing the tokenizer.json file.
	//
	// Example map: {"model-a": "/mnt/models/model-a/tokenizer.json", ...}
	ModelTokenizerMap map[string]string `json:"modelTokenizerMap,omitempty"`
}

// IsEnabled returns true if the local tokenizer configuration has any model mappings.
// A local tokenizer is considered enabled when at least one model-to-file mapping exists.
func (cfg *LocalTokenizerConfig) IsEnabled() bool {
	return cfg != nil && len(cfg.ModelTokenizerMap) > 0
}

// DefaultLocalTokenizerConfig creates a LocalTokenizerConfig by automatically discovering
// tokenizer files in the local tokenizer directory.
//
// Environment Variables:
//  1. LOCAL_TOKENIZER_DIR - base directory to search (defaults to /mnt/models)
//  2. LOCAL_TOKENIZER_FILENAME - tokenizer filename to look for (defaults to tokenizer.json)
//
// Auto-discovery Process:
//  1. Recursively walks the directory tree to find all tokenizer files
//  2. Attempts to parse HuggingFace cache directory structure first
//  3. Falls back to using relative path as model name for custom structures
//  4. Creates a mapping: model-name -> /path/to/tokenizer.json
//
// Supported directory structures:
//
//	HuggingFace cache (recommended):
//	  ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/{hash}/tokenizer.json
//	    -> "Qwen/Qwen3-0.6B"
//	  ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/{hash}/tokenizer.json
//	    -> "meta-llama/Llama-2-7b-chat-hf"
//	  ~/.cache/huggingface/hub/models--gpt2/snapshots/{hash}/tokenizer.json
//	    -> "gpt2"
//
//	Custom directory structures (arbitrary nesting):
//	  /mnt/models/llama-7b/tokenizer.json       -> "llama-7b"
//	  /mnt/models/Qwen/Qwen3/tokenizer.json     -> "Qwen/Qwen3"
//	  /mnt/models/a/b/c/tokenizer.json          -> "a/b/c"
//
// The model name is derived from the HuggingFace cache directory name when detected,
// otherwise it preserves the full directory structure relative to the base directory.
func DefaultLocalTokenizerConfig() (*LocalTokenizerConfig, error) {
	cfg := &LocalTokenizerConfig{
		AutoDiscoveryDir:               localTokenizerDir,
		AutoDiscoveryTokenizerFileName: localTokenizerFileName,
	}

	if err := discoverLocalTokenizerMap(cfg); err != nil {
		return nil, fmt.Errorf("failed to discover local tokenizer map: %w", err)
	}

	return cfg, nil
}

func discoverLocalTokenizerMap(cfg *LocalTokenizerConfig) error {
	if cfg == nil || cfg.AutoDiscoveryDir == "" || cfg.AutoDiscoveryTokenizerFileName == "" {
		return nil
	}

	if cfg.ModelTokenizerMap == nil {
		cfg.ModelTokenizerMap = make(map[string]string)
	}

	// Walk the directory tree recursively to find all tokenizer files
	err := filepath.WalkDir(cfg.AutoDiscoveryDir, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			//nolint:nilerr // Skip directories we can't read
			return nil
		}

		// Check if this is a tokenizer file
		if d.IsDir() || d.Name() != cfg.AutoDiscoveryTokenizerFileName {
			return nil
		}

		processTokenizerFile(cfg, path)
		return nil
	})
	// If the directory doesn't exist, that's okay - just return empty map
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("failed to walk LOCAL_TOKENIZER_DIR %q: %w", localTokenizerDir, err)
	}

	return nil
}

// processTokenizerFile processes a discovered tokenizer file and adds it to the model map.
func processTokenizerFile(cfg *LocalTokenizerConfig, path string) {
	// Get the full path up to the tokenizer file
	relPath, relErr := filepath.Rel(cfg.AutoDiscoveryDir, path)
	if relErr != nil {
		// Skip this file if we can't get relative path
		return
	}

	// Split the path into components
	pathComponents := strings.Split(relPath, string(filepath.Separator))

	var modelName string
	// Try to parse HuggingFace cache structure first
	// Look for a "models--*" directory in the path
	for _, component := range pathComponents {
		if parsedName, ok := parseHFCacheModelName(component); ok {
			modelName = parsedName
			break
		}
	}

	// If not HF cache structure, use the directory containing tokenizer.json
	if modelName == "" {
		modelDir := filepath.Dir(path)
		modelRelPath, err := filepath.Rel(cfg.AutoDiscoveryDir, modelDir)
		if err != nil {
			// Skip this file if we can't get relative path
			return
		}
		modelName = modelRelPath
	}

	// Only set model mappings that have not been overridden explicitly.
	if _, ok := cfg.ModelTokenizerMap[modelName]; !ok {
		cfg.ModelTokenizerMap[modelName] = path
	}
}

// parseHFCacheModelName extracts the model name from a HuggingFace cache directory path.
// HF cache directories follow the pattern: models--{org}--{model-name} or models--{model-name}
//
// Examples:
//   - "models--Qwen--Qwen3-0.6B" -> "Qwen/Qwen3-0.6B"
//   - "models--meta-llama--Llama-2-7b-chat-hf" -> "meta-llama/Llama-2-7b-chat-hf"
//   - "models--gpt2" -> "gpt2"
func parseHFCacheModelName(dirName string) (string, bool) {
	const prefix = "models--"
	if !strings.HasPrefix(dirName, prefix) {
		return "", false
	}

	// Remove the "models--" prefix and split by "--"
	remainder := strings.TrimPrefix(dirName, prefix)
	parts := strings.Split(remainder, "--")

	if len(parts) == 0 {
		return "", false
	}

	// Join parts with "/" to create HuggingFace-style model name
	return strings.Join(parts, "/"), true
}

// CachedTokenizer implements the Tokenizer interface for a specific model.
// It holds a single tokenizer instance that is initialized at creation time
// for the target model, providing efficient tokenization without caching overhead.
type CachedTokenizer struct {
	chatTemplateRenderer *preprocessing.ChatTemplatingProcessor
	tokenizerCacheKey    string
}

type HFCachedTokenizer struct {
	CachedTokenizer
	hfTokenizerConfig *HFTokenizerConfig
}
type LocalCachedTokenizer struct {
	CachedTokenizer
	localTokenizerConfig *LocalTokenizerConfig
}

// NewCachedHFTokenizer creates a new instance of CachedTokenizer downloading tokenizer configs from HuggingFace with
// the provided configuration.
func NewCachedHFTokenizer(ctx context.Context, modelName string, config *HFTokenizerConfig) (*HFCachedTokenizer, error) {
	chatTemplateRenderer := preprocessing.NewChatTemplatingProcessor()
	err := chatTemplateRenderer.Initialize()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize chat templater: %w", err)
	}

	tokenizerCacheKey, err := chatTemplateRenderer.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
		IsLocal:     false,
		Model:       modelName,
		DownloadDir: config.TokenizersCacheDir,
		Token:       config.HuggingFaceToken,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer with cache: %w", err)
	}

	return &HFCachedTokenizer{
		CachedTokenizer: CachedTokenizer{
			chatTemplateRenderer: chatTemplateRenderer,
			tokenizerCacheKey:    tokenizerCacheKey,
		},
		hfTokenizerConfig: config,
	}, nil
}

// NewCachedLocalTokenizer creates a new instance of CachedTokenizer that loads tokenizers
// from local files specified in the configuration.
//
// This is useful for:
//   - Air-gapped environments where HuggingFace is not accessible
//   - Pre-loaded models in containerized deployments
//   - Reducing startup latency by avoiding downloads
//
// The tokenizer is initialized for a specific model at creation time.
func NewCachedLocalTokenizer(ctx context.Context, modelName string, config LocalTokenizerConfig) (*LocalCachedTokenizer, error) {
	if err := discoverLocalTokenizerMap(&config); err != nil {
		return nil, fmt.Errorf("failed to discover local tokenizer map: %w", err)
	}

	chatTemplater := preprocessing.NewChatTemplatingProcessor()
	err := chatTemplater.Initialize()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize chat templater: %w", err)
	}

	path, ok := config.ModelTokenizerMap[modelName]
	if !ok {
		return nil, fmt.Errorf("tokenizer for model %q not found", modelName)
	}

	tokenizerCacheKey, err := chatTemplater.GetOrCreateTokenizerKey(ctx, &preprocessing.GetOrCreateTokenizerKeyRequest{
		IsLocal: true,
		Model:   path,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get or create tokenizer key with cache: %w", err)
	}

	return &LocalCachedTokenizer{
		CachedTokenizer: CachedTokenizer{
			chatTemplateRenderer: chatTemplater,
			tokenizerCacheKey:    tokenizerCacheKey,
		},
		localTokenizerConfig: &config,
	}, nil
}

func (t *CachedTokenizer) RenderChat(
	req *preprocessing.RenderChatRequest,
) ([]uint32, []preprocessing.Offset, error) {
	ctx := context.TODO()

	req.Key = t.tokenizerCacheKey
	tokens, offsets, err := t.chatTemplateRenderer.RenderChat(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to render chat template: %w", err)
	}

	return tokens, offsets, nil
}

// Render tokenizes the given prompt and returns token IDs with offset mappings.
func (t *CachedTokenizer) Render(prompt string) ([]uint32, []preprocessing.Offset, error) {
	ctx := context.TODO()

	tokens, offsets, err := t.chatTemplateRenderer.Render(ctx, &preprocessing.RenderRequest{
		Key:              t.tokenizerCacheKey,
		Text:             prompt,
		AddSpecialTokens: true,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to tokenize prompt: %w", err)
	}

	return tokens, offsets, nil
}

func (t *CachedTokenizer) Type() string {
	return "cached"
}

// getTokenizerCacheDir returns the absolute path to the tokenizer cache directory relative to the project root.
func getTokenizerCacheDir() string {
	if local := os.Getenv(localTokenizerDirEnv); local != "" {
		return local
	}

	_, filename, _, _ := runtime.Caller(0) // this file
	base := filepath.Dir(filename)
	return filepath.Join(base, "..", "..", "bin")
}

// CompositeTokenizer implements the Tokenizer interface with a fallback mechanism.
// It tries each tokenizer in order until one succeeds. This allows for graceful
// fallback from local tokenizers to HuggingFace tokenizers.
//
// Fallback behavior:
//  1. Tries the first tokenizer
//  2. If it fails, accumulates the error and tries the next
//  3. Returns immediately when a tokenizer succeeds
//  4. If all fail, returns all accumulated errors
//
// Example usage:
//
//	composite := &CompositeTokenizer{
//	    Tokenizers: []Tokenizer{
//	        localTokenizer,  // Try local first
//	        hfTokenizer,     // Fallback to HuggingFace
//	    },
//	}
//
// If the model exists locally, the local tokenizer is used. Otherwise, it falls back
// to downloading from HuggingFace. If all tokenizers fail, it returns a combined error.
type CompositeTokenizer struct {
	// Tokenizers is an ordered list of tokenizers to try.
	// They are attempted in order until one succeeds.
	Tokenizers []Tokenizer
}

func (c *CompositeTokenizer) RenderChat(
	req *preprocessing.RenderChatRequest,
) ([]uint32, []preprocessing.Offset, error) {
	var rErr error
	for _, tokenizer := range c.Tokenizers {
		copiedReq, err := req.DeepCopy()
		if err != nil {
			rErr = multierr.Append(rErr, fmt.Errorf("failed to copy chat render request: %w", err))
			continue
		}
		start := time.Now()
		ids, offsets, err := tokenizer.RenderChat(copiedReq)
		metrics.TokenizationLatency.WithLabelValues(tokenizer.Type()).Observe(time.Since(start).Seconds())
		if err != nil {
			rErr = multierr.Append(rErr, err)
			continue
		}
		metrics.TokenizedTokensCount.WithLabelValues(tokenizer.Type()).Add(float64(len(ids)))
		return ids, offsets, nil
	}
	return nil, nil, rErr
}

// Render tokenizes the given prompt and returns token IDs with offset mappings.
func (c *CompositeTokenizer) Render(prompt string,
) ([]uint32, []preprocessing.Offset, error) {
	var rErr error
	for _, tokenizer := range c.Tokenizers {
		start := time.Now()
		ids, offsets, err := tokenizer.Render(prompt)
		metrics.TokenizationLatency.WithLabelValues(tokenizer.Type()).Observe(time.Since(start).Seconds())
		if err != nil {
			rErr = multierr.Append(rErr, err)
			continue
		}
		metrics.TokenizedTokensCount.WithLabelValues(tokenizer.Type()).Add(float64(len(ids)))
		return ids, offsets, nil
	}
	return nil, nil, rErr
}

func (c *CompositeTokenizer) Type() string {
	return "composite"
}
