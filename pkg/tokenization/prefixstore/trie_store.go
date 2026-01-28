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

package prefixstore

import (
	"sync"

	preprocessing "github.com/llm-d/llm-d-kv-cache/pkg/preprocessing/chat_completions"
)

// containedTokenNode represents a node in the character-based Trie.
// TODO: consider chunking and hashing?
// It stores information about the last token fully contained within the prefix
// ending at this node.
type containedTokenNode struct {
	children map[rune]*containedTokenNode
	// lastContainedTokenID is that of the last token ending at or before this char position. 0 if none.
	lastContainedTokenID uint32
	// lastContainedTokenIndex is the index of that token in the original full list. -1 if none.
	lastContainedTokenIndex int
}

// newContainedTokenNode creates a new Trie node.
func newContainedTokenNode() *containedTokenNode {
	return &containedTokenNode{
		children:                make(map[rune]*containedTokenNode),
		lastContainedTokenIndex: -1,
		lastContainedTokenID:    0,
	}
}

// TrieTokenStore holds the root of the character-based prefix tree.
// TrieTokenStore is a character-based prefix tree that stores
// the last token fully contained within the prefix ending at each node.
type TrieTokenStore struct {
	mu   sync.RWMutex
	root *containedTokenNode
}

var _ Indexer = &TrieTokenStore{}

// AddTokenization adds the full tokenization of a string to the indexer.
// The function assumes tokens and offsets are of the same length.
// The function assumes that tokens will not be mutated after the call.
func (t *TrieTokenStore) AddTokenization(prompt string, tokens []uint32,
	offsets []preprocessing.Offset,
) error {
	if prompt == "" || len(tokens) == 0 || len(tokens) != len(offsets) {
		return nil
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	t.addFullTokenization(prompt, tokens, offsets)

	return nil
}

// NewContainedTokenTrie creates an empty containedTokenTrie.
func NewContainedTokenTrie() *TrieTokenStore {
	return &TrieTokenStore{
		root: newContainedTokenNode(),
	}
}

// addFullTokenization populates the Trie based on a fully tokenized string.
// It iterates through characters and determines the last contained token at
// each step.
// Assumes the caller holds the Write Lock.
func (t *TrieTokenStore) addFullTokenization(prompt string, tokens []uint32, offsets []preprocessing.Offset) {
	node := t.root
	var lastFoundK int
	if len(tokens) > 0 {
		t.root.lastContainedTokenIndex = 0
		t.root.lastContainedTokenID = tokens[0]
		lastFoundK = 0
	} else {
		lastFoundK = -1
	}

	for i, char := range prompt {
		//nolint:gosec // when models reach uint32 size of context window, none of us will be needed here
		charEndPos := uint(i) + 1

		// Find the largest token index 'k' such that offsets[k][1] <= charEndPos
		// We can continue searching forward from the previously found k
		currentBestK := lastFoundK
		searchStart := 0
		if lastFoundK != -1 {
			searchStart = lastFoundK // Start search from the last known good index
		}

		for k := searchStart; k < len(offsets); k++ {
			if offsets[k][1] <= charEndPos {
				if k > currentBestK { // Update if this index is greater
					currentBestK = k
				}
			} else {
				break
			}
		}
		lastFoundK = currentBestK

		// Traverse or create the node for the current character
		child, ok := node.children[char]
		if !ok {
			child = newContainedTokenNode()
			node.children[char] = child
		}
		node = child

		// Store the determined token info at this node
		if lastFoundK != -1 {
			node.lastContainedTokenIndex = lastFoundK
			node.lastContainedTokenID = tokens[lastFoundK]
		} else {
			node.lastContainedTokenIndex = -1
			node.lastContainedTokenID = 0
		}
	}
}

// FindLongestContainedTokens traverses the Trie for the prompt and collects
// the sequence of last contained tokens encountered.
//
//nolint:gocritic // unnamedResult: tokens and overlapRatio are self-explanatory from context
func (t *TrieTokenStore) FindLongestContainedTokens(prompt string) ([]uint32, float64) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	containedTokens := []uint32{}
	lastTokenIdxSeen := -1
	node := t.root

	// Handle potential token associated with the root (e.g., leading CLS)
	if node.lastContainedTokenIndex > lastTokenIdxSeen {
		containedTokens = append(containedTokens, node.lastContainedTokenID)
		lastTokenIdxSeen = node.lastContainedTokenIndex
	}

	overlapRatio := 0.0
	for i, char := range prompt {
		child, ok := node.children[char]
		if !ok {
			break
		}
		node = child

		// Check if this node represents a *new* contained token compared to the sequence so far
		if node.lastContainedTokenIndex > lastTokenIdxSeen {
			containedTokens = append(containedTokens, node.lastContainedTokenID)
			lastTokenIdxSeen = node.lastContainedTokenIndex
		}

		overlapRatio = float64(i+1) / float64(len(prompt))
	}

	return containedTokens, overlapRatio
}
