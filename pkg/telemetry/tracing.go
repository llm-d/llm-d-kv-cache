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

// Package telemetry provides OpenTelemetry tracing initialization for standalone kv-cache services.
//
// IMPORTANT: When llm-d-kv-cache is used as a library (e.g., bundled into llm-d-router),
// the library code uses otel.Tracer() directly to access the global tracer provider initialized by
// the parent application. This package is only used for standalone examples and services.
package telemetry

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strconv"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.24.0"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/klog/v2"

	"github.com/llm-d/llm-d-kv-cache/version"
)

const (
	// defaultServiceName is the default OpenTelemetry service name.
	// Can be overridden via OTEL_SERVICE_NAME environment variable.
	defaultServiceName = "llm-d-kv-cache"

	// InstrumentationName identifies this instrumentation library in traces.
	InstrumentationName = "llm-d-kv-cache"
)

// stripScheme removes the scheme from an endpoint URL, returning host:port.
// This is required for gRPC clients that expect host:port format only.
func stripScheme(endpoint string) string {
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" {
		return endpoint // not a valid URL, return as-is
	}
	return u.Host
}

// InitTracing initializes OpenTelemetry tracing.
// Configuration is done via environment variables:
// - OTEL_SERVICE_NAME: Service name (default: llm-d-kv-cache)
// - OTEL_TRACES_EXPORTER: Span exporter, "otlp", "console" or "none" (default: otlp)
// - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)
// - OTEL_TRACES_SAMPLER: Sampling strategy (default: parentbased_traceidratio)
// - OTEL_TRACES_SAMPLER_ARG: Sampling ratio (default: 0.1 for 10%).
func InitTracing(ctx context.Context) (func(context.Context) error, error) {
	logger := klog.FromContext(ctx)

	// Get service name from environment
	serviceName := os.Getenv("OTEL_SERVICE_NAME")
	if serviceName == "" {
		serviceName = defaultServiceName
	}
	// Get OTLP endpoint from environment and strip scheme
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		endpoint = "localhost:4317"
	}
	endpoint = stripScheme(endpoint)

	// Get sampling ratio from environment
	samplingRatio := 0.1 // default 10%
	if ratioStr := os.Getenv("OTEL_TRACES_SAMPLER_ARG"); ratioStr != "" {
		if ratio, err := strconv.ParseFloat(ratioStr, 64); err == nil {
			samplingRatio = ratio
		} else {
			logger.Error(err, "Invalid OTEL_TRACES_SAMPLER_ARG, using default", "value", ratioStr, "default", 0.1)
		}
	}

	exporterType := traceExporterType(logger)

	logger.Info("Initializing OpenTelemetry tracing",
		"endpoint", endpoint,
		"service", serviceName,
		"exporter", exporterType,
		"samplingRatio", samplingRatio)

	// Create resource with service name and build version
	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion(version.BuildRef),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create resource: %w", err)
	}

	opt := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.TraceIDRatioBased(samplingRatio))),
	}

	// "none" registers no span processor at all. Spans are still created and
	// propagated, so instrumented code and context propagation are unaffected.
	if exporterType != exporterTypeNone {
		exporter, err := newSpanExporter(ctx, exporterType, endpoint)
		if err != nil {
			return nil, fmt.Errorf("failed to create trace exporter: %w", err)
		}
		opt = append(opt, sdktrace.WithBatcher(exporter))
	}

	// Create trace provider with parent-based sampling
	tp := sdktrace.NewTracerProvider(opt...)

	// Set global trace provider
	otel.SetTracerProvider(tp)

	// Set W3C trace context propagator
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	// Route OpenTelemetry's internal errors to the structured logger.
	otel.SetErrorHandler(&errorHandler{logger: logger})

	logger.Info("OpenTelemetry tracing initialized successfully")

	// Return shutdown function
	return tp.Shutdown, nil
}

// Tracer returns a tracer for the given instrumentation scope, defaulting to
// InstrumentationName. Build version and commit SHA are attached so every span
// in a trace carries consistent scope metadata. When used as a library, the
// host application's tracer provider determines the service name.
func Tracer(scope ...string) trace.Tracer {
	name := InstrumentationName
	if len(scope) > 0 && scope[0] != "" {
		name = scope[0]
	}
	return otel.Tracer(
		name,
		trace.WithInstrumentationVersion(version.BuildRef),
		trace.WithInstrumentationAttributes(
			attribute.String("commit-sha", version.CommitSHA),
		),
	)
}

// The exporter types OTEL_TRACES_EXPORTER selects between.
const (
	exporterTypeOTLP    = "otlp"
	exporterTypeConsole = "console"
	exporterTypeNone    = "none"

	defaultExporterType = exporterTypeOTLP
)

// traceExporterType resolves OTEL_TRACES_EXPORTER to one of the types
// newSpanExporter builds:
//
//   - otlp: export spans through gRPC to an opentelemetry collector
//   - console: pretty print spans on stdout, for development
//   - none: create spans but export nothing
//
// An unrecognised value falls back to otlp with a logged warning rather than
// failing startup; the exporter is not worth failing over.
func traceExporterType(logger logr.Logger) string {
	exporterType := os.Getenv("OTEL_TRACES_EXPORTER")
	if exporterType == "" {
		return defaultExporterType
	}

	switch exporterType {
	case exporterTypeOTLP, exporterTypeConsole, exporterTypeNone:
		return exporterType
	default:
		logger.Info("Unsupported OTEL_TRACES_EXPORTER, falling back to otlp", "value", exporterType)
		return defaultExporterType
	}
}

// newSpanExporter builds the exporter named by exporterType, which traceExporterType
// has already narrowed to otlp or console; "none" is handled by the caller and never
// reaches here.
func newSpanExporter(ctx context.Context, exporterType, endpoint string) (sdktrace.SpanExporter, error) {
	if exporterType == exporterTypeConsole {
		return stdouttrace.New(stdouttrace.WithPrettyPrint())
	}

	return otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(), // Use WithTLSCredentials() in production
	)
}

// errorHandler routes OpenTelemetry's internal errors to a structured logger
// instead of the process stderr.
type errorHandler struct {
	logger logr.Logger
}

func (h *errorHandler) Handle(err error) {
	h.logger.Error(err, "OpenTelemetry trace error")
}
