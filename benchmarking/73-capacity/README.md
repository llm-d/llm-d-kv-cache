# Inference-Perf Benchmark Report

### Workload profile

```yaml
# X-70R-6kSys — 8×307,328 KV; target 70%-80% total usage
#
# CLUSTER
# - Pods: 8
# - KV/pod: 307,328 → Cluster capacity C_cap = 2,458,624
#
# SHAPE
# - system_prompt_len = 6,000
# - question_len      = 1,200      # ≤ 1,500; cached per user
# - num_prompts_per_group = 5      # users per group
# - output_len        = 1,000      # observed live ≈ 200–230 tokens/running request
#
# RESIDENT SIZING
# - Resident per group R_group = 6,000 + 5×1,200 = 12,000
# - num_groups G = 150
#   Resident total = 150 × 12,000 = 1,800,000 → 1,800,000 / 2,458,624 ≈ 73.2%
#
# LIVE SIZING
#
# WORKING SET
# - One FULL SET S = G × P = 150 × 5 = 750 requests
#
load:
  type: poisson
  stages:
    # Warmup — seat residents (~1×S)
    - rate: 15
      duration: 50            # 15*50  = 750

    # Main ladder
    - rate: 3
      duration: 20
    - rate: 10
      duration: 20
    - rate: 15
      duration: 20
    - rate: 20
      duration: 38           
    - rate: 22
      duration: 34           
    - rate: 25
      duration: 30           
    - rate: 30
      duration: 25           
    - rate: 35
      duration: 21           
    - rate: 40               
      duration: 38           
    - rate: 43
      duration: 36           
    - rate: 46
      duration: 33           
    - rate: 49
      duration: 30           
    - rate: 52
      duration: 29           
    - rate: 55
      duration: 27           
    - rate: 57
      duration: 26           
    - rate: 60
      duration: 25           
    
api:
  type: completion
  streaming: true

server:
  type: vllm
  model_name: Qwen/Qwen3-32B
  base_url: <endpoint>
  ignore_eos: true

tokenizer:
  pretrained_model_name_or_path: Qwen/Qwen3-32B

data:
  type: shared_prefix
  shared_prefix:
    num_groups: 150
    num_prompts_per_group: 5
    system_prompt_len: 6000
    question_len: 1200
    output_len: 1000

report:
  request_lifecycle:
    summary: true
    per_stage: true
    per_request: true

storage:
  local_storage:
    path: /workspace
```

### Scheduler Configurations

**estimated-scheduling** (matching the default configuration in the IGW)

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: single-profile-handler
  - type: prefix-cache-scorer
    parameters:
      hashBlockSize: 64
      maxPrefixBlocksToMatch: 256
      lruCapacityPerServer: 31250
  - type: kv-cache-scorer
  - type: queue-scorer
  - type: max-score-picker
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: kv-cache-scorer
        weight: 1.0
      - pluginRef: queue-scorer
        weight: 1.0
      - pluginRef: prefix-cache-scorer
        weight: 1.0
      - pluginRef: max-score-picker
```

**load-scheduling**

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: queue-scorer
- type: kv-cache-scorer
- type: max-score-picker
  parameters:
    maxNumOfEndpoints: 1
- type: single-profile-handler
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: queue-scorer
    weight: 1
  - pluginRef: kv-cache-scorer
    weight: 1
  - pluginRef: max-score-picker
```

**precise-scheduling**

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: single-profile-handler
- type: prefix-cache-scorer
  parameters:
    mode: cache_tracking
    tokenProcessorConfig:
      blockSize: 64   
      hashSeed: "42"
    indexerConfig:
      kvBlockIndexConfig:
        enableMetrics: true    
        metricsLoggingInterval: 60000000000 
- type: kv-cache-scorer
- type: queue-scorer
- type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
    - pluginRef: prefix-cache-scorer
      weight: 3.0
    - pluginRef: kv-cache-scorer
      weight: 2.0
    - pluginRef: queue-scorer
      weight: 2.0
    - pluginRef: max-score-picker
```

**random-scheduling**

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: single-profile-handler
- type: random-picker
schedulingProfiles:
- name: default
  plugins:
    - pluginRef: random-picker
```

## Charts

### Latency vs QPS

<img src="latency_vs_qps.png" alt="Latency vs QPS" width="720"/>

### Throughput vs QPS

<img src="throughput_vs_qps.png" alt="Throughput vs QPS" width="720"/>

### TTFT p90 vs QPS

<img src="ttft_p90_vs_qps.png" alt="TTFT p90 vs QPS" width="300"/>

### Waiting Queue vs Time

<img src="waiting_queue_vs_time.png" alt="Waiting Queue vs Time" width="720"/>

### KV Cache Usage vs Time

<img src="kv_cache_usage_vs_time.png" alt="KV Cache Usage vs Time" width="720"/>

### EPP Metrics Comparative Analysis

<img src="epp_metrics_comparison.png" alt="EPP Metrics Comparative Analysis" width="720"/>

### How to read this report (quick)

- **Output tokens/sec** is the primary throughput metric (higher is better).

- **Requests/sec** shows the rate of completed requests.

- **Success Rate** reflects outcome quality, not volume.

- **TTFT** is time to first token; **ITL** is the gap between tokens (both lower is better).

- **Queue sizes** and **KV cache usage** show resource utilization patterns.


### Summary across QPS


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|-----------:|---:|---:|---:|---:|
| precise-scheduling | 8730.0 |     35.562 | 0.542 | 0.298 | 0.026 | 0.0000/0.063 |
| estimated-scheduling | 6944.4 |     36.068 | 31.083 | 13.316 | 0.024 | 0.0000/0.044 |
| random-scheduling | 4428.7 |     35.671 | 92.551 | 45.281 | 0.025 | 0.0000/0.036 |
| load-scheduling | 4266.4 |     36.298 | 94.865 | 46.987 | 0.024 | 0.0000/0.036 |

### EPP Queue and KV Cache Metrics Summary


| Experiment | Wait Queue (mean/p90/max) | KV Cache % (mean/p90/max) | Pods | Data Points |
|---|---:|---:|---:|---:|
| estimated-scheduling | 8.1/33/65 | 66.2/99.7/100.0 | 8 | 206912 |
| load-scheduling | 28.9/67/124 | 54.3/98.6/100.0 | 8 | 209104 |
| precise-scheduling | 0.1/0/39 | 59.3/86.5/100.0 | 8 | 208032 |
| random-scheduling | 27.3/63/108 | 59.0/98.9/100.0 | 8 | 207536 |

## Per-QPS Results


### QPS = 3.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| estimated-scheduling | 1310.9 | 3.093 | 0.497 | 0.300 | 0.011 | 0.0000/0.023 |
| precise-scheduling | 1219.1 | 2.620 | 0.498 | 0.291 | 0.011 | 0.0000/0.022 |
| random-scheduling | 1314.0 | 2.752 | 0.820 | 0.622 | 0.012 | 0.0000/0.023 |
| load-scheduling | 1274.8 | 2.997 | 1.609 | 0.944 | 0.012 | 0.0000/0.023 |

### QPS = 10.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| estimated-scheduling | 3842.4 | 9.386 | 0.499 | 0.271 | 0.013 | 0.0000/0.029 |
| precise-scheduling | 3902.1 | 9.529 | 0.600 | 0.335 | 0.013 | 0.0000/0.028 |
| random-scheduling | 3568.5 | 9.564 | 1.398 | 0.826 | 0.016 | 0.0000/0.030 |
| load-scheduling | 3607.0 | 10.469 | 4.124 | 2.241 | 0.016 | 0.0000/0.030 |

### QPS = 15.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| estimated-scheduling | 5152.9 | 13.642 | 0.400 | 0.232 | 0.015 | 0.0000/0.033 |
| precise-scheduling | 5761.1 | 15.544 | 0.400 | 0.249 | 0.015 | 0.0000/0.032 |
| random-scheduling | 3791.1 | 16.484 | 2.299 | 1.278 | 0.020 | 0.0000/0.036 |
| load-scheduling | 3348.4 | 15.152 | 7.002 | 4.013 | 0.020 | 0.0000/0.035 |

### QPS = 20.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| estimated-scheduling | 8082.6 | 19.593 | 0.600 | 0.329 | 0.021 | 0.0000/0.057 |
| precise-scheduling | 9184.5 | 20.159 | 0.600 | 0.363 | 0.021 | 0.0000/0.048 |
| random-scheduling | 4933.1 | 18.838 | 70.273 | 25.073 | 0.024 | 0.0000/0.036 |
| load-scheduling | 4902.5 | 22.157 | 77.707 | 29.229 | 0.024 | 0.0000/0.036 |

### QPS = 22.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 9382.0 | 22.003 | 0.500 | 0.300 | 0.020 | 0.0000/0.049 |
| estimated-scheduling | 8415.7 | 23.832 | 0.700 | 0.407 | 0.022 | 0.0000/0.048 |
| load-scheduling | 4785.5 | 21.469 | 72.972 | 28.710 | 0.024 | 0.0000/0.036 |
| random-scheduling | 4765.4 | 21.986 | 78.420 | 28.185 | 0.024 | 0.0000/0.036 |

### QPS = 25.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 9242.2 | 24.268 | 0.583 | 0.303 | 0.021 | 0.0000/0.050 |
| estimated-scheduling | 8445.1 | 24.510 | 0.609 | 0.370 | 0.022 | 0.0000/0.048 |
| random-scheduling | 4781.3 | 23.400 | 82.943 | 30.689 | 0.024 | 0.0000/0.036 |
| load-scheduling | 4872.8 | 25.597 | 85.979 | 32.534 | 0.024 | 0.0000/0.036 |

### QPS = 30.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 9221.8 | 30.992 | 0.509 | 0.296 | 0.021 | 0.0000/0.049 |
| estimated-scheduling | 8435.9 | 30.242 | 0.708 | 0.373 | 0.023 | 0.0000/0.048 |
| load-scheduling | 4888.7 | 29.711 | 83.806 | 32.831 | 0.024 | 0.0000/0.036 |
| random-scheduling | 4636.3 | 30.368 | 86.182 | 32.212 | 0.024 | 0.0000/0.036 |

### QPS = 35.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 9780.6 | 32.861 | 0.500 | 0.275 | 0.021 | 0.0000/0.048 |
| estimated-scheduling | 7443.5 | 35.050 | 2.459 | 2.780 | 0.026 | 0.0000/0.045 |
| load-scheduling | 4660.7 | 35.167 | 88.086 | 33.874 | 0.024 | 0.0000/0.036 |
| random-scheduling | 4332.6 | 36.403 | 91.519 | 34.043 | 0.024 | 0.0000/0.036 |

### QPS = 40.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10270.8 | 40.431 | 0.488 | 0.269 | 0.027 | 0.0000/0.095 |
| estimated-scheduling | 8045.9 | 40.212 | 43.018 | 13.160 | 0.027 | 0.0000/0.046 |
| load-scheduling | 4946.4 | 39.019 | 100.911 | 53.545 | 0.025 | 0.0000/0.036 |
| random-scheduling | 4773.8 | 41.192 | 101.272 | 54.196 | 0.025 | 0.0000/0.036 |

### QPS = 43.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10054.8 | 40.521 | 0.500 | 0.281 | 0.028 | 0.0000/0.091 |
| estimated-scheduling | 7969.6 | 43.120 | 43.515 | 16.941 | 0.026 | 0.0000/0.043 |
| random-scheduling | 4844.6 | 42.775 | 101.142 | 53.247 | 0.025 | 0.0000/0.036 |
| load-scheduling | 4973.3 | 40.143 | 101.524 | 53.802 | 0.025 | 0.0000/0.036 |

### QPS = 46.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10128.7 | 44.630 | 0.589 | 0.293 | 0.028 | 0.0000/0.086 |
| estimated-scheduling | 7695.2 | 45.986 | 44.930 | 18.081 | 0.026 | 0.0000/0.044 |
| random-scheduling | 4969.5 | 47.578 | 101.265 | 53.572 | 0.025 | 0.0000/0.036 |
| load-scheduling | 4723.6 | 46.688 | 105.884 | 56.746 | 0.025 | 0.0000/0.036 |

### QPS = 49.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10296.5 | 48.942 | 0.600 | 0.311 | 0.029 | 0.0000/0.062 |
| estimated-scheduling | 7455.9 | 48.461 | 46.335 | 19.111 | 0.026 | 0.0000/0.043 |
| random-scheduling | 4783.2 | 48.305 | 103.514 | 56.130 | 0.025 | 0.0000/0.036 |
| load-scheduling | 4004.8 | 49.162 | 105.015 | 55.208 | 0.025 | 0.0000/0.036 |

### QPS = 52.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10284.6 | 52.371 | 0.507 | 0.288 | 0.029 | 0.0000/0.062 |
| estimated-scheduling | 7452.3 | 52.162 | 47.284 | 21.940 | 0.026 | 0.0000/0.042 |
| random-scheduling | 4845.1 | 51.544 | 105.004 | 56.035 | 0.025 | 0.0000/0.036 |
| load-scheduling | 3991.0 | 54.050 | 108.572 | 58.807 | 0.025 | 0.0000/0.036 |

### QPS = 55.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10140.9 | 54.410 | 0.600 | 0.308 | 0.029 | 0.0000/0.061 |
| estimated-scheduling | 7100.7 | 55.860 | 50.108 | 22.327 | 0.026 | 0.0000/0.041 |
| random-scheduling | 4921.7 | 53.552 | 104.321 | 54.844 | 0.025 | 0.0000/0.036 |
| load-scheduling | 4671.9 | 54.779 | 110.681 | 58.467 | 0.025 | 0.0000/0.036 |

### QPS = 57.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10313.9 | 57.655 | 0.600 | 0.319 | 0.029 | 0.0000/0.061 |
| estimated-scheduling | 7013.8 | 56.362 | 49.930 | 23.423 | 0.026 | 0.0000/0.042 |
| random-scheduling | 4764.2 | 55.068 | 106.816 | 56.803 | 0.025 | 0.0000/0.036 |
| load-scheduling | 3866.5 | 55.837 | 113.278 | 61.750 | 0.025 | 0.0000/0.036 |

### QPS = 60.0


| Experiment | Output toks/s | Requests/s | TTFT p90 (s) | TTFT mean (s) | ITL mean (s) | ITL p50/ p90 (s) |
|---|---:|---:|---:|---:|---:|---:|
| precise-scheduling | 10496.9 | 59.242 | 0.500 | 0.283 | 0.029 | 0.0000/0.061 |
| estimated-scheduling | 7247.9 | 60.508 | 50.226 | 24.281 | 0.026 | 0.0000/0.042 |
| random-scheduling | 4835.2 | 56.912 | 105.988 | 57.426 | 0.025 | 0.0000/0.036 |
| load-scheduling | 4745.1 | 59.971 | 110.184 | 58.497 | 0.025 | 0.0000/0.036 |

## Per-Pod EPP Metrics

Individual pod metrics over the duration of each experiment.


### Experiment: estimated-scheduling

**Pod:** `estimated-scheduling_10_132_1_19`

<img src="epp_pod_estimated-scheduling_10_132_1_19.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_132_1_20`

<img src="epp_pod_estimated-scheduling_10_132_1_20.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_132_2_205`

<img src="epp_pod_estimated-scheduling_10_132_2_205.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_132_2_206`

<img src="epp_pod_estimated-scheduling_10_132_2_206.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_133_1_2`

<img src="epp_pod_estimated-scheduling_10_133_1_2.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_133_1_3`

<img src="epp_pod_estimated-scheduling_10_133_1_3.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_133_3_179`

<img src="epp_pod_estimated-scheduling_10_133_3_179.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>

**Pod:** `estimated-scheduling_10_135_1_172`

<img src="epp_pod_estimated-scheduling_10_135_1_172.png" alt="Per-pod metrics for estimated-scheduling" width="1440"/>


### Experiment: load-scheduling

**Pod:** `load-scheduling_10_132_1_19`

<img src="epp_pod_load-scheduling_10_132_1_19.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_132_1_20`

<img src="epp_pod_load-scheduling_10_132_1_20.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_132_2_205`

<img src="epp_pod_load-scheduling_10_132_2_205.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_132_2_206`

<img src="epp_pod_load-scheduling_10_132_2_206.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_133_1_2`

<img src="epp_pod_load-scheduling_10_133_1_2.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_133_1_3`

<img src="epp_pod_load-scheduling_10_133_1_3.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_133_3_179`

<img src="epp_pod_load-scheduling_10_133_3_179.png" alt="Per-pod metrics for load-scheduling" width="1440"/>

**Pod:** `load-scheduling_10_135_1_172`

<img src="epp_pod_load-scheduling_10_135_1_172.png" alt="Per-pod metrics for load-scheduling" width="1440"/>


### Experiment: precise-scheduling

**Pod:** `precise-scheduling_10_132_1_19`

<img src="epp_pod_precise-scheduling_10_132_1_19.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_132_1_20`

<img src="epp_pod_precise-scheduling_10_132_1_20.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_132_2_205`

<img src="epp_pod_precise-scheduling_10_132_2_205.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_132_2_206`

<img src="epp_pod_precise-scheduling_10_132_2_206.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_133_1_2`

<img src="epp_pod_precise-scheduling_10_133_1_2.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_133_1_3`

<img src="epp_pod_precise-scheduling_10_133_1_3.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_133_3_179`

<img src="epp_pod_precise-scheduling_10_133_3_179.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>

**Pod:** `precise-scheduling_10_135_1_172`

<img src="epp_pod_precise-scheduling_10_135_1_172.png" alt="Per-pod metrics for precise-scheduling" width="1440"/>


### Experiment: random-scheduling

**Pod:** `random-scheduling_10_132_1_19`

<img src="epp_pod_random-scheduling_10_132_1_19.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_132_1_20`

<img src="epp_pod_random-scheduling_10_132_1_20.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_132_2_205`

<img src="epp_pod_random-scheduling_10_132_2_205.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_132_2_206`

<img src="epp_pod_random-scheduling_10_132_2_206.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_133_1_2`

<img src="epp_pod_random-scheduling_10_133_1_2.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_133_1_3`

<img src="epp_pod_random-scheduling_10_133_1_3.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_133_3_179`

<img src="epp_pod_random-scheduling_10_133_3_179.png" alt="Per-pod metrics for random-scheduling" width="1440"/>

**Pod:** `random-scheduling_10_135_1_172`

<img src="epp_pod_random-scheduling_10_135_1_172.png" alt="Per-pod metrics for random-scheduling" width="1440"/>
