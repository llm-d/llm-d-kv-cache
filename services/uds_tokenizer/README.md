# UDS Tokenizer Service

This service provides tokenization functionality via gRPC over Unix Domain Socket (UDS). It also exposes a separate HTTP endpoint for Kubernetes health checks.

## Features

- Apply chat templates to messages
- Tokenize text prompts
- Runtime configuration updates
- Health check endpoint for Kubernetes
- Support for multiple model formats (HuggingFace, ModelScope)
- Automatic model downloading and caching
- gRPC-based communication for efficient tokenization

## Services

The service exposes gRPC methods over UDS and HTTP endpoints for health/config:

1. `TokenizationService.Tokenize` - Tokenize text via gRPC (UDS only)
2. `TokenizationService.RenderChatTemplate` - Apply chat template via gRPC (UDS only)
3. `/health` - Health check endpoint (TCP port, for Kubernetes probes)
4. `/config` - Get or update configuration (TCP port)

## Quick Start

Start the service:
```bash
python run_grpc_server.py
```

The service will:
- Listen on `/tmp/tokenizer/tokenizer-uds.socket` for gRPC calls
- Listen on port 8080 (configurable via PROBE_PORT) for health checks

## Environment Variables

| Variable | Description | Default |
|---------|-------------|---------|
| `LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `THREAD_POOL_SIZE` | Number of worker threads for all CPU-intensive operations | 2 * CPU cores (limited by container resources, max 32) |
| `MODEL` | Path to the model directory | ./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| `ADD_SPECIAL_TOKENS` | Whether to add special tokens | true |
| `ENABLE_THINKING` | Whether to enable thinking mode | false |
| `ADD_GENERATION_PROMPT` | Whether to add generation prompt | true |
| `PROBE_PORT` | Port for health check endpoint | 8082 |
| `USE_MODELSCOPE` | Whether to download tokenizer files from ModelScope (true) or Hugging Face (false) | false |

## gRPC Service Definition

The service implements the `TokenizationService` defined in `tokenizer.proto`:

```protobuf
service TokenizationService {
  // Tokenize converts a text input to token IDs
  rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);

  // RenderChatTemplate renders a chat template with the given messages
  rpc RenderChatTemplate(ChatTemplateRequest) returns (ChatTemplateResponse);
}
```

### Tokenize Method

Converts text input to token IDs.

Request:
- `input`: Text to tokenize
- `model_name`: Model name (currently not used in implementation)
- `add_special_tokens`: Whether to add special tokens

Response:
- `input_ids`: List of token IDs
- `offset_pairs`: Flattened array of [start, end, start, end, ...] character offsets
- `success`: Whether the request was successful
- `error_message`: Error message if the request failed

### RenderChatTemplate Method

Renders a chat template with the given messages.

Request:
- `messages`: List of messages with role and content
- `chat_template`: Chat template to use
- `add_generation_prompt`: Whether to add generation prompt
- Other template-specific parameters

Response:
- `rendered_prompt`: The rendered chat template
- `success`: Whether the request was successful
- `error_message`: Error message if the request failed

## HTTP Endpoints

### GET /health
Health check endpoint for Kubernetes probes.

Response:
```json
{
  "status": "healthy",
  "service": "tokenizer-service",
  "timestamp": 1234567890.123
}
```

### GET /config
Get current configuration.

Response:
```json
{
  "model": "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
  "add_special_tokens": true,
  "enable_thinking": false,
  "add_generation_prompt": true
}
```

### POST /config
Update configuration at runtime.

Request body:
```json
{
  "model": "./models/qwen/qwen3-0.6b",
  "add_special_tokens": false
}
```

Response:
```json
{
  "status": "success",
  "message": "Configuration updated successfully"
}
```

## Testing

### Unit Tests

Run unit tests with mocks (no service needed):
```bash
# Install test dependencies
pip install -r tests/requirements.txt

# Run unit tests
python -m pytest tests/test_tokenizer_unit.py -v
```

### Integration Tests

Run integration tests (requires service to be running):
```bash
# Start the service in the background
python run_grpc_server.py &

# Run integration tests with automatic waiting
python tests/run_integration_tests.py

# Stop the service
pkill -f "python run_grpc_server.py"
```

The integration test runner will automatically wait for the server to be ready before running tests.

## Kubernetes Deployment

The service is designed to run in Kubernetes with:
- A shared `emptyDir` volume for UDS communication between containers
- Health check endpoint for liveness and readiness probes
- Proper security context with non-root user

## Model Support

The service supports:
- HuggingFace models (local or remote)
- ModelScope models (automatically downloaded and cached)
- Custom models in standard format

Models are automatically downloaded and cached in the `models/` directory.
The source for downloading can be controlled with the `USE_MODELSCOPE` environment variable:
- `false` (default): Download from Hugging Face
- `true`: Download from ModelScope

See [models/README.md](models/README.md) for detailed information about model caching, pre-populating the cache, and Kubernetes deployment strategies.

## Project Structure

```
├── run_grpc_server.py       # Main gRPC server entry point
├── tokenizer_grpc_service.py # gRPC service implementation
├── tokenizer_service/       # Core tokenizer service implementation
│   ├── __init__.py
│   ├── tokenizer.py         # Tokenizer service implementation
│   └── exceptions.py        # Custom exceptions
├── tokenizerpb/              # gRPC service definition
│   ├── tokenizer_pb2_grpc.py
│   └── tokenizer_pb2.py
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── logger.py            # Logger functionality
├── tests/                   # Test files
│   ├── __init__.py
│   ├── run_integration_tests.py  # Integration test runner
│   └── test_tokenizer_unit.py    # Unit tests
├── models/                  # Model files (downloaded automatically)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```