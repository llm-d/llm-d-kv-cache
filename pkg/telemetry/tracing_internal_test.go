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

package telemetry

import (
	"context"
	"fmt"
	"testing"

	"github.com/go-logr/logr"
)

func TestTraceExporterType(t *testing.T) {
	tests := []struct {
		name string
		env  string
		set  bool
		want string
	}{
		{name: "unset defaults to otlp", set: false, want: exporterTypeOTLP},
		{name: "otlp", env: "otlp", set: true, want: exporterTypeOTLP},
		{name: "console", env: "console", set: true, want: exporterTypeConsole},
		{name: "none", env: "none", set: true, want: exporterTypeNone},
		{name: "an unrecognised value falls back to otlp", env: "jaeger", set: true, want: exporterTypeOTLP},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.set {
				t.Setenv("OTEL_TRACES_EXPORTER", tc.env)
			}

			got := traceExporterType(logr.Discard())
			if got != tc.want {
				t.Errorf("traceExporterType() = %q, want %q", got, tc.want)
			}
		})
	}
}

// newSpanExporter must build exactly the exporter it was asked for. The stdout
// exporter in particular must not be constructed for the otlp type.
func TestNewSpanExporter(t *testing.T) {
	tests := []struct {
		exporterType string
		wantType     string
	}{
		{exporterType: exporterTypeOTLP, wantType: "*otlptrace.Exporter"},
		{exporterType: exporterTypeConsole, wantType: "*stdouttrace.Exporter"},
	}

	for _, tc := range tests {
		t.Run(tc.exporterType, func(t *testing.T) {
			exporter, err := newSpanExporter(context.Background(), tc.exporterType, "localhost:4317")
			if err != nil {
				t.Fatalf("newSpanExporter(%q) error = %v", tc.exporterType, err)
			}
			t.Cleanup(func() {
				if err := exporter.Shutdown(context.Background()); err != nil {
					t.Errorf("exporter.Shutdown() error = %v", err)
				}
			})

			if got := fmt.Sprintf("%T", exporter); got != tc.wantType {
				t.Errorf("newSpanExporter(%q) = %s, want %s", tc.exporterType, got, tc.wantType)
			}
		})
	}
}

// "none" registers no exporter and InitTracing must not call newSpanExporter for it,
// so span creation, sampling and propagation stay intact while nothing is exported.
func TestNoneExporterStillCreatesSpans(t *testing.T) {
	t.Setenv("OTEL_TRACES_EXPORTER", "none")
	t.Setenv("OTEL_TRACES_SAMPLER_ARG", "1")

	shutdown, err := InitTracing(context.Background())
	if err != nil {
		t.Fatalf("InitTracing() error = %v", err)
	}
	t.Cleanup(func() {
		if err := shutdown(context.Background()); err != nil {
			t.Errorf("shutdown() error = %v", err)
		}
	})

	_, span := Tracer().Start(context.Background(), "none-exporter-span")
	defer span.End()

	if !span.SpanContext().IsValid() {
		t.Error("span context is not valid, want spans to still be created and propagated")
	}
	if !span.SpanContext().IsSampled() {
		t.Error("span is not sampled, want the sampler to be unaffected by the exporter")
	}
}
