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
- Initialize without pre-loading a specific model
- Listen on `/tmp/tokenizer/tokenizer-uds.socket` for gRPC calls
- Listen on port 8082 (configurable via PROBE_PORT) for health checks

Before using tokenization methods, initialize the tokenizer for a specific model using the `InitializeTokenizer` gRPC method.

## Environment Variables

| Variable | Description | Default |
|---------|-------------|---------|
| `LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `THREAD_POOL_SIZE` | Number of worker threads for all CPU-intensive operations | 2 * CPU cores (limited by container resources, max 32) |
| `PROBE_PORT` | Port for health check endpoint | 8082 |
| `USE_MODELSCOPE` | Whether to download tokenizer files from ModelScope (true) or Hugging Face (false) | false |
| `ENABLE_GRPC_REFLECTION` | Enable gRPC server reflection for service discovery | disabled |


## gRPC Service Definition

The service implements the `TokenizationService` defined in `tokenizer.proto`:

```protobuf
service TokenizationService {
  // Tokenize converts a text input to token IDs
  rpc Tokenize(TokenizeRequest) returns (TokenizeResponse);

  // RenderChatTemplate renders a chat template with the given messages
  rpc RenderChatTemplate(ChatTemplateRequest) returns (ChatTemplateResponse);

  // InitializeTokenizer initializes the tokenizer for a specific model
  rpc InitializeTokenizer(InitializeTokenizerRequest) returns (InitializeTokenizerResponse);
}
```

### Tokenize Method

Converts text input to token IDs.

Request:
- `input`: Text to tokenize
- `model_name`: Model name to use for tokenization
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
- `model_name`: Model name to use for applying the template
- Other template-specific parameters

Response:
- `rendered_prompt`: The rendered chat template
- `success`: Whether the request was successful
- `error_message`: Error message if the request failed

## Additional gRPC Methods

### InitializeTokenizer Method

Initializes the tokenizer for a specific model.

Request:
- `model_name`: Model name to initialize the tokenizer for
- `enable_thinking`: Whether to enable thinking tokens
- `add_generation_prompt`: Whether to add generation prompt

Response:
- `success`: Whether the initialization was successful
- `error_message`: Error message if the initialization failed

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

## Usage Examples

You can interact with the gRPC service using `grpcurl`.

**Note:** gRPC reflection must be enabled for `grpcurl` to work. Set the environment variable before starting the server:
```bash
export ENABLE_GRPC_REFLECTION=1
python run_grpc_server.py
```

Reflection is disabled by default for security reasons, as it increases the exposed surface area by allowing service/method/message discovery.

### List available services
```bash
grpcurl -plaintext unix:///tmp/tokenizer/tokenizer-uds.socket list
```
```
grpc.reflection.v1alpha.ServerReflection
tokenization.TokenizationService
```

### Describe the TokenizationService
```bash
grpcurl -plaintext unix:///tmp/tokenizer/tokenizer-uds.socket describe tokenization.TokenizationService
```
```
tokenization.TokenizationService is a service:
service TokenizationService {
  rpc InitializeTokenizer ( .tokenization.InitializeTokenizerRequest ) returns ( .tokenization.InitializeTokenizerResponse );
  rpc RenderChatTemplate ( .tokenization.ChatTemplateRequest ) returns ( .tokenization.ChatTemplateResponse );
  rpc Tokenize ( .tokenization.TokenizeRequest ) returns ( .tokenization.TokenizeResponse );
}
```

### Initialize tokenizer for a specific model
```bash
grpcurl -plaintext -d '{"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}' \
  unix:///tmp/tokenizer/tokenizer-uds.socket \
  tokenization.TokenizationService/InitializeTokenizer
```
```json
{
  "success": true
}
```

### Tokenize text
```bash
grpcurl -plaintext -d '{
  "input": "Hello world",
  "add_special_tokens": true,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}' unix:///tmp/tokenizer/tokenizer-uds.socket \
  tokenization.TokenizationService/Tokenize
```
```json
{
  "input_ids": [
    9707,
    1879
  ],
  "success": true,
  "offset_pairs": [
    0,
    5,
    5,
    11
  ]
}
```

### Render chat template
```bash
grpcurl -plaintext -d '{
  "conversation_turns": [{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }],
  "add_generation_prompt": true,
  "model_name": "Qwen/Qwen2.5-0.5B-Instruct"
}' unix:///tmp/tokenizer/tokenizer-uds.socket \
  tokenization.TokenizationService/RenderChatTemplate
```
```json
{
  "rendered_prompt": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello, how are you?<|im_end|>\n",
  "success": true
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