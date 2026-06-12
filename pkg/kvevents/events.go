// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents

// KVCacheSpecKind identifies vLLM KV cache group semantics. The string values
// mirror vLLM's KVCacheSpec.kind wire values, parsed from the engine event.
//
// Only the kinds the scorer classifies on are enumerated. vLLM's other kinds
// (mamba, chunked_local_attention, cross_attention, encoder_only_attention)
// flow through as neither main attention nor sliding window, so no named
// constants are needed for them.
type KVCacheSpecKind string

// EventType represents the type of KV-cache event.
type EventType string

const (
	// EventTypeBlockStored indicates blocks were added to cache.
	EventTypeBlockStored EventType = "BlockStored"
	// EventTypeBlockRemoved indicates blocks were evicted from cache.
	EventTypeBlockRemoved EventType = "BlockRemoved"
	// EventTypeAllBlocksCleared indicates entire cache was cleared.
	EventTypeAllBlocksCleared EventType = "AllBlocksCleared"
)

const (
	KVCacheSpecKindFullAttention     KVCacheSpecKind = "full_attention"
	KVCacheSpecKindMLAAttention      KVCacheSpecKind = "mla_attention"
	KVCacheSpecKindSlidingWindow     KVCacheSpecKind = "sliding_window"
	KVCacheSpecKindSlidingWindowMLA  KVCacheSpecKind = "sliding_window_mla"
	KVCacheSpecKindSinkFullAttention KVCacheSpecKind = "sink_full_attention"
)

// IsMainAttention reports whether the kind is a "main attention" group
// (full / MLA / sink-full). In vLLM's hybrid KV cache the realized prefix-cache
// hit converges to the minimum across groups, and the main-attention group is
// the binding constraint: it requires the whole prefix to be cached, whereas
// sliding-window and other groups are looser. Scoring uses main-attention
// groups to define the candidate prefix.
func (k KVCacheSpecKind) IsMainAttention() bool {
	switch k {
	case KVCacheSpecKindFullAttention, KVCacheSpecKindMLAAttention, KVCacheSpecKindSinkFullAttention:
		return true
	default:
		return false
	}
}

// IsSlidingWindow reports whether the kind is a sliding-window group, which
// needs only a trailing window of contiguous cached blocks for a hit.
func (k KVCacheSpecKind) IsSlidingWindow() bool {
	return k == KVCacheSpecKindSlidingWindow || k == KVCacheSpecKindSlidingWindowMLA
}

// GenericEvent represents a KV-cache event containing already-parsed data.
type GenericEvent interface {
	// Type returns the event type.
	Type() EventType
}

// EventBatch represents a batch of generic events from an inference engine.
type EventBatch struct {
	Timestamp float64
	Events    []GenericEvent
}

// RawMessage holds the raw transport-level data from a received pub/sub message.
// It contains no domain-specific fields — parsing is deferred to the EngineAdapter.
type RawMessage struct {
	// Topic is the original transport topic string.
	Topic string
	// Sequence is the message sequence number from the transport.
	Sequence uint64
	// Payload is the raw encoded event batch bytes, not yet decoded.
	Payload []byte
}

// EngineAdapter defines the interface for engine-specific message parsers.
// Each inference engine has its own adapter implementation that handles
// parsing raw transport messages into domain events.
type EngineAdapter interface {
	// ParseMessage parses a raw transport message into domain data.
	// It extracts pod identity and model name from the topic,
	// and decodes the payload into an EventBatch.
	ParseMessage(msg *RawMessage) (podID, modelName string, batch EventBatch, err error)

	// ShardingKey extracts the key used to shard messages across worker queues.
	// Messages with the same sharding key are guaranteed to be processed in order.
	ShardingKey(msg *RawMessage) string
}

// BlockStoredEvent represents blocks being added to the cache.
type BlockStoredEvent struct {
	BlockHashes []uint64
	Tokens      []uint32
	ParentHash  uint64
	BlockSize   int
	DeviceTier  string
	LoraID      *int
	LoraName    *string
	ExtraKeys   [][]any
	// GroupIdx identifies the vLLM KV cache group that emitted this event.
	GroupIdx *int
	// KVCacheSpecKind carries vLLM's semantic cache type for the group.
	KVCacheSpecKind KVCacheSpecKind
	// KVCacheSpecSlidingWindowSize carries the SWA window size when applicable.
	KVCacheSpecSlidingWindowSize *int
}

// Type returns the event type.
func (e *BlockStoredEvent) Type() EventType {
	return EventTypeBlockStored
}

// BlockRemovedEvent represents blocks being evicted from the cache.
type BlockRemovedEvent struct {
	BlockHashes []uint64
	DeviceTier  string
	// GroupIdx identifies the vLLM KV cache group that removed this block.
	GroupIdx *int
}

// Type returns the event type.
func (e *BlockRemovedEvent) Type() EventType {
	return EventTypeBlockRemoved
}

// AllBlocksClearedEvent represents all blocks being cleared from a pod's cache.
type AllBlocksClearedEvent struct {
	DeviceTier string
}

// Type returns the event type.
func (e *AllBlocksClearedEvent) Type() EventType {
	return EventTypeAllBlocksCleared
}
