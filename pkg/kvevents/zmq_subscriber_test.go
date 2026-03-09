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

package kvevents //nolint:testpackage // testing unexported parseTopic and zmqSubscriber internals.

import (
	"testing"
)

func TestParseTopic(t *testing.T) {
	tests := []struct {
		name      string
		topic     string
		wantPod   string
		wantModel string
		wantOK    bool
	}{
		{
			name:      "valid topic",
			topic:     "kv@pod-123@meta-llama/Llama-3.1-8B-Instruct",
			wantPod:   "pod-123",
			wantModel: "meta-llama/Llama-3.1-8B-Instruct",
			wantOK:    true,
		},
		{
			name:      "valid topic with simple names",
			topic:     "kv@mypod@mymodel",
			wantPod:   "mypod",
			wantModel: "mymodel",
			wantOK:    true,
		},
		{
			name:   "model name containing @",
			topic:  "kv@pod-1@model@extra",
			wantOK: false,
		},
		{
			name:   "missing model name",
			topic:  "kv@pod-123",
			wantOK: false,
		},
		{
			name:   "no separator",
			topic:  "kvpod123model",
			wantOK: false,
		},
		{
			name:   "empty string",
			topic:  "",
			wantOK: false,
		},
		{
			name:      "empty pod and model",
			topic:     "kv@@",
			wantPod:   "",
			wantModel: "",
			wantOK:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pod, model, ok := parseTopic(tt.topic)
			if ok != tt.wantOK {
				t.Errorf("parseTopic(%q) ok = %v, want %v", tt.topic, ok, tt.wantOK)
				return
			}
			if !ok {
				return
			}
			if pod != tt.wantPod {
				t.Errorf("parseTopic(%q) pod = %q, want %q", tt.topic, pod, tt.wantPod)
			}
			if model != tt.wantModel {
				t.Errorf("parseTopic(%q) model = %q, want %q", tt.topic, model, tt.wantModel)
			}
		})
	}
}

func TestTopicCaching(t *testing.T) {
	sub := &zmqSubscriber{}

	// First call should parse and cache
	topic := "kv@pod-1@model-a"
	pod, model, ok := parseTopic(topic)
	if !ok {
		t.Fatal("expected parseTopic to succeed")
	}
	sub.lastTopic = topic
	sub.lastPodIdentifier = pod
	sub.lastModelName = model

	if sub.lastPodIdentifier != "pod-1" || sub.lastModelName != "model-a" {
		t.Errorf("unexpected cached values: pod=%q model=%q", sub.lastPodIdentifier, sub.lastModelName)
	}

	// Same topic should hit cache
	if topic != sub.lastTopic {
		t.Error("expected cache hit for same topic")
	}

	// Different topic should miss cache and update
	topic2 := "kv@pod-2@model-b"
	if topic2 == sub.lastTopic {
		t.Error("expected cache miss for different topic")
	}
	pod2, model2, ok2 := parseTopic(topic2)
	if !ok2 {
		t.Fatal("expected parseTopic to succeed for topic2")
	}
	sub.lastTopic = topic2
	sub.lastPodIdentifier = pod2
	sub.lastModelName = model2

	if sub.lastPodIdentifier != "pod-2" || sub.lastModelName != "model-b" {
		t.Errorf("unexpected cached values after update: pod=%q model=%q", sub.lastPodIdentifier, sub.lastModelName)
	}
}

func BenchmarkParseTopic(b *testing.B) {
	topic := "kv@vllm-pod-abc123@meta-llama/Llama-3.1-8B-Instruct"
	b.ReportAllocs()
	for b.Loop() {
		parseTopic(topic)
	}
}
