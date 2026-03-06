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
package events

// BlockRemovedEvent represents blocks being evicted from the cache.
// All hashes are already parsed to uint64 by the adapter.
type BlockRemovedEvent struct {
	BlockHashes []uint64
	DeviceTier  string
}

// Type returns the event type.
func (e *BlockRemovedEvent) Type() EventType {
	return EventTypeBlockRemoved
}
