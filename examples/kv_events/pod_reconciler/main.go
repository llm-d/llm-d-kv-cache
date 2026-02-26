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
	"os"
	"os/signal"
	"syscall"

	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		cancel()
	}()

	// Setup logger
	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))
	logger := log.FromContext(ctx)

	if err := run(ctx); err != nil {
		logger.Error(err, "Failed to run pod reconciler example")
		os.Exit(1) //nolint:gocritic
	}
}

func run(ctx context.Context) error {
	logger := log.FromContext(ctx)

	// Create a scheme for controller-runtime
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		return err
	}

	// Create a controller-runtime manager
	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		return err
	}

	// Setup KV Cache Index
	indexConfig := kvblock.DefaultIndexConfig()
	index, err := kvblock.NewIndex(ctx, indexConfig)
	if err != nil {
		logger.Error(err, "failed to create index")
		return err
	}

	// Setup event pool
	poolConfig := kvevents.DefaultConfig()
	tokenProcessor := kvblock.NewChunkedTokenDatabase(kvblock.DefaultTokenProcessorConfig())
	pool := kvevents.NewPool(poolConfig, index, tokenProcessor)
	pool.Start(ctx)

	// Create subscriber manager
	subscriberManager := kvevents.NewSubscriberManager(pool)

	// Convert to internal reconciler config
	reconcilerConfig, err := NewPodReconcilerConfig(
		poolConfig.PodDiscoveryConfig,
		poolConfig.TopicFilter,
	)
	if err != nil {
		logger.Error(err, "failed to create reconciler config")
		return err
	}

	// Create and register the pod reconciler
	podReconciler := &PodReconciler{
		Client:            mgr.GetClient(),
		Scheme:            mgr.GetScheme(),
		Config:            reconcilerConfig,
		SubscriberManager: subscriberManager,
	}

	if err := podReconciler.SetupWithManager(mgr); err != nil {
		logger.Error(err, "failed to setup pod reconciler")
		return err
	}

	logger.Info("=== Pod Reconciler Example Started ===")
	logger.Info("Watching pods with label selector", "selector", poolConfig.PodDiscoveryConfig.PodLabelSelector)
	logger.Info("Topic filter", "filter", poolConfig.TopicFilter)

	// Start the manager (this will start the reconciler)
	mgrCtx, mgrCancel := context.WithCancel(ctx)
	defer mgrCancel()

	go func() {
		if err := mgr.Start(mgrCtx); err != nil {
			logger.Error(err, "failed to start manager")
		}
	}()

	// Wait for shutdown
	<-ctx.Done()
	logger.Info("Shutting down pod reconciler example...")

	// Shutdown subscriber manager
	subscriberManager.Shutdown(ctx)

	// Shutdown pool
	pool.Shutdown(ctx)

	logger.Info("Pod reconciler example shut down successfully")
	return nil
}
