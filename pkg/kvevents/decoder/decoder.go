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

// Decoder defines the interface for encoding and decoding raw bytes.
type Decoder interface {
	// Decode unmarshals data into the provided value.
	Decode(data []byte, v interface{}) error

	// Encode marshals the provided value into bytes.
	Encode(v interface{}) ([]byte, error)
}
