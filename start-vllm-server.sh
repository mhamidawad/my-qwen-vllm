#!/bin/bash
# vLLM Server Startup Script
# Usage: ./start-vllm-server.sh [gpu-memory-utilization] [max-batch-tokens]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
GPU_MEMORY_UTIL=${1:-0.8}
MAX_BATCH_TOKENS=${2:-8192}
PORT=8000
MODEL="Qwen/Qwen3-0.6B"
ADAPTER_NAME="my-qwen-adapter"

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}   Starting vLLM Server with Qwen LoRA Adapter${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Adapter: $ADAPTER_NAME"
echo "  GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "  Max Batch Tokens: $MAX_BATCH_TOKENS"
echo "  Port: $PORT"
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM not installed${NC}"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check if adapter files exist
if [ ! -f "adapter_config.json" ]; then
    echo -e "${RED}Error: adapter_config.json not found in current directory${NC}"
    exit 1
fi

echo -e "${GREEN}Starting vLLM server...${NC}"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-lora \
  --lora-modules "$ADAPTER_NAME=./" \
  --max-lora-rank 16 \
  --lora-extra-vocab-size 256 \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --max-model-len 8192 \
  --max-num-batched-tokens "$MAX_BATCH_TOKENS" \
  --max-num-seqs 256 \
  --dtype auto \
  --use-v2-block-manager \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --worker-use-ray false

# If we reach here, the server stopped
echo -e "${YELLOW}Server stopped${NC}"
