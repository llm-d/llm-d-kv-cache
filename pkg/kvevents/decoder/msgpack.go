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

package decoder

import (
	"fmt"

	"github.com/vmihailenco/msgpack/v5"
)

// MsgpackDecoder implements Decoder for MessagePack format.
type MsgpackDecoder struct{}

// NewMsgpackDecoder creates a new msgpack decoder.
func NewMsgpackDecoder() *MsgpackDecoder {
	return &MsgpackDecoder{}
}

// Decode unmarshals msgpack data into the provided value.
func (m *MsgpackDecoder) Decode(data []byte, v interface{}) error {
	if err := msgpack.Unmarshal(data, v); err != nil {
		return fmt.Errorf("failed to decode msgpack: %w", err)
	}
	return nil
}

// Encode marshals the provided value into msgpack bytes.
func (m *MsgpackDecoder) Encode(v interface{}) ([]byte, error) {
	data, err := msgpack.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("failed to encode msgpack: %w", err)
	}
	return data, nil
}
