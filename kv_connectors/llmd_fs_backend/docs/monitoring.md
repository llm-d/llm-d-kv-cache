# Monitoring vLLM with KV Offload Metrics

End-to-end guide: deploy vLLM + Prometheus + Grafana in K8s, port-forward to view dashboards locally, and validate with a benchmark.

## 1. Deploy vLLM with the FS Backend

```bash
# Create HF token secret and PVCs
export HF_TOKEN=<your-token>
kubectl create secret generic hf-token --from-literal=HF_TOKEN="$HF_TOKEN"
kubectl apply -f docs/deployment/vllm-pvc.yaml

# Deploy vLLM
kubectl apply -f docs/deployment/vllm-storage.yaml
```

Wait for the pod to become ready (model download + startup takes a few minutes):

```bash
kubectl wait --for=condition=ready pod -l app=vllm-storage --timeout=600s
```

## 2. Deploy Prometheus and Grafana

```bash
# Prometheus — scrapes vLLM metrics every 15s
kubectl apply -f docs/deployment/monitoring/prometheus.yaml

# Grafana dashboard ConfigMap (must be applied before Grafana)
kubectl apply -f docs/deployment/monitoring/grafana-dashboard-configmap.yaml

# Grafana — pre-configured with Prometheus datasource and the dashboard
kubectl apply -f docs/deployment/monitoring/grafana.yaml
```

## 3. Port-Forward to Your Machine

Open two terminals:

```bash
# Terminal 1: Grafana UI on http://localhost:3000
kubectl port-forward svc/grafana-svc 3000:3000
```

```bash
# Terminal 2: Prometheus UI on http://localhost:9090 (optional, for ad-hoc queries)
kubectl port-forward svc/prometheus-svc 9090:9090
```

Open the dashboard directly at:

**http://localhost:3000/d/vllm-kv-offload/vllm-kv-offload-dashboard**

Anonymous access is enabled so no login is needed.

## 4. Run a Benchmark

Port-forward the vLLM service and run the benchmark:

```bash
# Terminal 3: vLLM API on http://localhost:8000
kubectl port-forward svc/vllm-storage-svc 8000:8000
```

### Benchmark with Prefix Repetition (KV Cache Offload)

Run two benchmark iterations to test KV cache offload (write) and retrieval (read):

**Run 1: KV Cache Write/Offload Test**
```bash
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-32B \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 16384 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 100 \
  --prefix-repetition-output-len 5 \
  --num-prompts 100 \
  --max-concurrency 40 \
  --request-rate 40 \
  --burstiness 1 \
  --ignore-eos \
  --seed 42
```

**Run 2: KV Cache Read/Retrieval Test**
```bash
vllm bench serve \
  --backend vllm \
  --base-url http://localhost:8000 \
  --model Qwen/Qwen3-32B \
  --dataset-name prefix_repetition \
  --prefix-repetition-prefix-len 16384 \
  --prefix-repetition-suffix-len 0 \
  --prefix-repetition-num-prefixes 100 \
  --prefix-repetition-output-len 5 \
  --num-prompts 100 \
  --max-concurrency 40 \
  --request-rate 40 \
  --burstiness 1 \
  --ignore-eos \
  --seed 42
```

Watch the Grafana dashboard — you should see KV offload metrics (throughput, transfer rates, bytes offloaded) once the cache starts spilling to storage during the benchmark.

## 5. Verify Metrics Manually

```bash
curl -s http://localhost:8000/metrics | grep kv_offload
```

## 6. Cleanup

Remove all monitoring and vLLM resources:

```bash
# Monitoring stack
kubectl delete -f docs/deployment/monitoring/grafana.yaml
kubectl delete -f docs/deployment/monitoring/grafana-dashboard-configmap.yaml
kubectl delete -f docs/deployment/monitoring/prometheus.yaml

# vLLM
kubectl delete -f docs/deployment/vllm-storage.yaml
kubectl delete -f docs/deployment/vllm-pvc.yaml
kubectl delete secret hf-token
```
