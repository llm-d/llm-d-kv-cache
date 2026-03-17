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

package helper

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"sync/atomic"

	zmq4 "github.com/go-zeromq/zmq4"
	"github.com/vmihailenco/msgpack/v5"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// Publisher sends KV cache events to a ZMQ endpoint.
type Publisher struct {
	socket   zmq4.Socket
	endpoint string
	seqNum   uint64
}

// NewPublisher creates a new ZMQ publisher.
// endpoint is the ZMQ address to connect to (e.g., "tcp://localhost:5557").
func NewPublisher(ctx context.Context, endpoint string) (*Publisher, error) {
	socket := zmq4.NewPub(ctx)

	if err := socket.Dial(endpoint); err != nil {
		socket.Close()
		return nil, fmt.Errorf("failed to connect to %s: %w", endpoint, err)
	}

	return &Publisher{
		socket:   socket,
		endpoint: endpoint,
	}, nil
}

// PublishEvent publishes a KV cache event batch to the ZMQ topic.
// topic should include the pod identifier (e.g., "kv.pod1").
func (p *Publisher) PublishEvent(ctx context.Context, topic string, batch interface{}) error {
	logger := log.FromContext(ctx).V(0)

	// Use an encoder configured for struct as array
	var payload bytes.Buffer
	enc := msgpack.NewEncoder(&payload)
	enc.UseArrayEncodedStructs(true)
	err := enc.Encode(batch)
	if err != nil {
		return fmt.Errorf("failed to marshal event batch: %w", err)
	}

	// sequence number for ordering
	seq := atomic.AddUint64(&p.seqNum, 1)
	seqBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seqBytes, seq)

	// send topic, sequence, payload as a 3-frame message
	msg := zmq4.NewMsgFrom([]byte(topic), seqBytes, payload.Bytes())
	if err := p.socket.Send(msg); err != nil {
		return fmt.Errorf("failed to send message to topic %s: %w", topic, err)
	}

	logger.Info("Published event batch", "topic", topic, "seq", seq)
	return nil
}

// Close closes the publisher and cleans up resources.
func (p *Publisher) Close() error {
	if p.socket != nil {
		return p.socket.Close()
	}
	return nil
}

func SetupPublisher(ctx context.Context) (*Publisher, error) {
	logger := log.FromContext(ctx)

	endpoint := "tcp://localhost:5557"
	logger.Info("Creating ZMQ publisher (simulating vLLM engines)", "endpoint", endpoint)

	publisher, err := NewPublisher(ctx, endpoint)
	if err != nil {
		return nil, fmt.Errorf("failed to create ZMQ publisher: %w", err)
	}

	logger.Info("ZMQ publisher created successfully")
	return publisher, nil
}
