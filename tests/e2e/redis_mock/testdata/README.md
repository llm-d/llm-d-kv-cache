# Test data

## Download model metadata

```shell
 hf download --exclude "*safetensors" \
  --local-dir ./tests/e2e/redis_mock/testdata/local-llama3 \
  RedHatAI/Meta-Llama-3-8B-Instruct-FP8
```