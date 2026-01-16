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

package main

import (
	"context"
	"fmt"
	"net"
	"strings"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

// PodReconcilerConfig holds the runtime configuration for the pod reconciler.
// This is the internal configuration used by the reconciler with parsed selectors.
type PodReconcilerConfig struct {
	// PodLabelSelector is the parsed label selector to filter which pods to watch.
	PodLabelSelector labels.Selector
	// PodNamespace limits watching to a specific namespace. Empty means all namespaces.
	PodNamespace string
	// TopicFilter is the ZMQ subscription filter (e.g., "kv@").
	TopicFilter string
	// SocketPort is the port where vLLM pods expose ZMQ (default: 5557).
	SocketPort string
}

// NewPodReconcilerConfig creates a PodReconcilerConfig from kvevents.PodDiscoveryConfig.
func NewPodReconcilerConfig(cfg *kvevents.PodDiscoveryConfig, topicFilter string) (*PodReconcilerConfig, error) {
	if cfg == nil {
		cfg = kvevents.DefaultPodReconcilerConfig()
	}

	// Parse label selector
	selector, err := labels.Parse(cfg.PodLabelSelector)
	if err != nil {
		return nil, fmt.Errorf("failed to parse pod label selector: %w", err)
	}

	// Set defaults
	socketPort := cfg.SocketPort
	if socketPort == 0 {
		socketPort = 5557
	}

	return &PodReconcilerConfig{
		PodLabelSelector: selector,
		PodNamespace:     cfg.PodNamespace,
		TopicFilter:      topicFilter,
		SocketPort:       fmt.Sprintf("%d", socketPort),
	}, nil
}

// PodReconciler watches pods and manages per-pod ZMQ subscribers.
type PodReconciler struct {
	client.Client
	Scheme            *runtime.Scheme
	Config            *PodReconcilerConfig
	SubscriberManager *kvevents.SubscriberManager
}

// Reconcile handles pod events and manages ZMQ subscribers.
func (r *PodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.Info("Reconciling pod", "pod", req.NamespacedName)

	var pod corev1.Pod
	if err := r.Get(ctx, req.NamespacedName, &pod); err != nil {
		if errors.IsNotFound(err) {
			// Pod was deleted, remove the subscriber
			debugLogger.Info("Pod deleted, removing subscriber", "pod", req)
			r.SubscriberManager.RemoveSubscriber(ctx, req.String())
			return ctrl.Result{}, nil
		}
		debugLogger.Error(err, "Failed to get pod")
		return ctrl.Result{}, err
	}

	// Check if pod matches our label selector
	if !r.Config.PodLabelSelector.Matches(labels.Set(pod.Labels)) {
		debugLogger.Info("Pod does not match label selector, skipping", "pod", req)
		return ctrl.Result{}, nil
	}

	// Check if pod is in a state where we should subscribe
	if shouldSubscribe := r.shouldSubscribeToPod(&pod); !shouldSubscribe {
		// Pod is not ready, remove subscriber if it exists
		debugLogger.Info("Pod not in subscribable state, removing subscriber if exists",
			"pod", req.NamespacedName, "phase", pod.Status.Phase)
		r.SubscriberManager.RemoveSubscriber(ctx, req.String())
		return ctrl.Result{}, nil
	}

	// Pod is ready, ensure we have a subscriber
	podIdentifier := req.String()
	endpoint := r.buildEndpoint(&pod)

	debugLogger.Info("Ensuring subscriber for pod",
		"pod", req,
		"endpoint", endpoint,
		"podIP", pod.Status.PodIP)

	if err := r.SubscriberManager.EnsureSubscriber(ctx, podIdentifier, endpoint, r.Config.TopicFilter, true); err != nil {
		debugLogger.Error(err, "Failed to ensure subscriber for pod", "pod", req)
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// shouldSubscribeToPod determines if a pod is in a state where we should subscribe to it.
func (r *PodReconciler) shouldSubscribeToPod(pod *corev1.Pod) bool {
	// Pod must be running and have an IP
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}

	if pod.Status.PodIP == "" {
		return false
	}

	// Check if pod is ready
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}

	return false
}

// buildEndpoint constructs the ZMQ endpoint from the pod information.
func (r *PodReconciler) buildEndpoint(pod *corev1.Pod) string {
	return "tcp://" + net.JoinHostPort(strings.TrimSpace(pod.Status.PodIP), r.Config.SocketPort)
}

// SetupWithManager sets up the controller with the Manager.
func (r *PodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	// Create a predicate to filter pod events
	podPredicate := predicate.NewPredicateFuncs(func(object client.Object) bool {
		pod, ok := object.(*corev1.Pod)
		if !ok {
			return false
		}

		// Filter by namespace if configured
		if r.Config.PodNamespace != "" && pod.Namespace != r.Config.PodNamespace {
			return false
		}

		// Filter by label selector
		return r.Config.PodLabelSelector.Matches(labels.Set(pod.Labels))
	})

	builder := ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		WithEventFilter(podPredicate)

	// If namespace is specified, watch only that namespace
	// This is handled by the predicate above, but we can optimize by
	// configuring the manager's cache to watch specific namespace
	// (requires manager-level configuration)

	return builder.Complete(r)
}
