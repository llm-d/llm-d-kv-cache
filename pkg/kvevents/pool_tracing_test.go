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

package kvevents //nolint:testpackage // tests use unexported processRawMessage

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

// fakeAdapter is a test EngineAdapter that returns canned parse results.
type fakeAdapter struct {
	podID     string
	modelName string
	batch     EventBatch
	err       error
}

//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func (f *fakeAdapter) ParseMessage(*RawMessage) (string, string, EventBatch, error) {
	return f.podID, f.modelName, f.batch, f.err
}

func (f *fakeAdapter) ShardingKey(*RawMessage) string { return f.podID }

// recordSpans installs a span recorder as the global tracer provider and
// returns it so emitted spans can be inspected.
func recordSpans(t *testing.T) *tracetest.SpanRecorder {
	t.Helper()
	recorder := tracetest.NewSpanRecorder()
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder)))
	return recorder
}

// spanByName returns the first recorded span with the given name.
func spanByName(spans []sdktrace.ReadOnlySpan, name string) (sdktrace.ReadOnlySpan, bool) {
	for _, s := range spans {
		if s.Name() == name {
			return s, true
		}
	}
	return nil, false
}

// TestProcessRawMessage_EmitsSpans verifies that the event-processing path emits
// a parent "process" span and a child "decode" span, that the decode span nests
// under the process span, and that key attributes are recorded.
func TestProcessRawMessage_EmitsSpans(t *testing.T) {
	recorder := recordSpans(t)

	pool, _, _ := newTestPool(t, 4)
	pool.adapter = &fakeAdapter{podID: "pod-1", modelName: "llama"}

	msg := &RawMessage{Topic: "kv@pod-1@llama", Sequence: 7, Payload: []byte("payload-bytes")}
	pool.processRawMessage(context.Background(), msg)

	spans := recorder.Ended()

	process, ok := spanByName(spans, "events_process")
	require.True(t, ok, "expected a process span")
	decode, ok := spanByName(spans, "events_decode")
	require.True(t, ok, "expected a decode span")

	// decode must be a child of process.
	assert.Equal(t, process.SpanContext().SpanID(), decode.Parent().SpanID(),
		"decode span should be a child of the process span")

	attrs := map[string]string{}
	ints := map[string]int64{}
	for _, kv := range process.Attributes() {
		switch kv.Value.Type().String() {
		case "STRING":
			attrs[string(kv.Key)] = kv.Value.AsString()
		case "INT64":
			ints[string(kv.Key)] = kv.Value.AsInt64()
		}
	}
	assert.Equal(t, "kv@pod-1@llama", attrs["llm_d.kv_cache.events.topic"])
	assert.Equal(t, int64(7), ints["llm_d.kv_cache.events.sequence"])
	assert.Equal(t, int64(len("payload-bytes")), ints["llm_d.kv_cache.events.payload_size_bytes"])
}

// TestProcessRawMessage_DecodeErrorSetsStatus verifies that a decode failure
// marks both the decode and process spans with an error status.
func TestProcessRawMessage_DecodeErrorSetsStatus(t *testing.T) {
	recorder := recordSpans(t)

	pool, _, _ := newTestPool(t, 4)
	pool.adapter = &fakeAdapter{err: errors.New("boom")}

	pool.processRawMessage(context.Background(), &RawMessage{Topic: "kv@pod-1"})

	spans := recorder.Ended()

	process, ok := spanByName(spans, "events_process")
	require.True(t, ok)
	decode, ok := spanByName(spans, "events_decode")
	require.True(t, ok)

	assert.Equal(t, codes.Error, process.Status().Code)
	assert.Equal(t, codes.Error, decode.Status().Code)
}
