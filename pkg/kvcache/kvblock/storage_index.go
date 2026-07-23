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

package kvblock

import (
	"encoding/binary"
	"sync"

	cuckoo "github.com/seiflotfy/cuckoofilter"
)

const (
	defaultFilterCapacity uint = 10_000_000

	// SharedStorageBackendName is the normalized device tier/backend name used
	// for shared-storage KV blocks.
	SharedStorageBackendName = "shared_storage"
	// ObjectStoreBackendName is the normalized device tier name used for object
	// store KV blocks.
	ObjectStoreBackendName = "object_store"
)

// StorageIndexConfig holds the configuration for the storage checkpoint index.
type StorageIndexConfig struct {
	Enabled        bool `json:"enabled"`
	FilterCapacity uint `json:"filterCapacity"`
}

// DefaultStorageIndexConfig returns a default configuration for the storage index.
func DefaultStorageIndexConfig() *StorageIndexConfig {
	return &StorageIndexConfig{
		Enabled:        false,
		FilterCapacity: defaultFilterCapacity,
	}
}

// StorageIndex tracks sparse checkpoint hashes that exist on storage.
// Unlike the GPU/CPU Index, checkpoints are global (no pod qualifier), and
// block hashes are already canonical request-key checkpoints (no engine-key
// resolution). Cuckoo-backed implementations are approximate: lookups may
// return false positives, and deletion can be affected by fingerprint
// collisions.
type StorageIndex interface {
	// AddCheckpoint returns true when the checkpoint is present after the call.
	// It returns false only when the backing filter cannot insert it.
	AddCheckpoint(requestKey BlockHash) bool
	HasCheckpoint(requestKey BlockHash) bool
	RemoveCheckpoint(requestKey BlockHash)
	Clear()
	SetStride(stride int)
	Stride() int
}

type cuckooStorageIndex struct {
	filter *cuckoo.Filter
	mu     sync.RWMutex
	stride int
}

// NewCuckooStorageIndex creates a new StorageIndex backed by a Cuckoo filter.
func NewCuckooStorageIndex(capacity uint) StorageIndex {
	if capacity == 0 {
		capacity = defaultFilterCapacity
	}
	return &cuckooStorageIndex{
		filter: cuckoo.NewFilter(capacity),
	}
}

func checkpointKey(requestKey BlockHash) []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(requestKey))
	return buf
}

func (c *cuckooStorageIndex) AddCheckpoint(requestKey BlockHash) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	key := checkpointKey(requestKey)
	if c.filter.Lookup(key) {
		return true
	}
	return c.filter.Insert(key)
}

func (c *cuckooStorageIndex) HasCheckpoint(requestKey BlockHash) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.filter.Lookup(checkpointKey(requestKey))
}

func (c *cuckooStorageIndex) RemoveCheckpoint(requestKey BlockHash) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.filter.Delete(checkpointKey(requestKey))
}

func (c *cuckooStorageIndex) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.filter.Reset()
}

func (c *cuckooStorageIndex) SetStride(stride int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.stride = stride
}

func (c *cuckooStorageIndex) Stride() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.stride
}
