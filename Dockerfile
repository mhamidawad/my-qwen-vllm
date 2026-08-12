# Multi-stage build for optimized production image
FROM vllm/vllm-openai:latest as builder

# Stage 1: Runtime environment
FROM vllm/vllm-openai:latest

LABEL maintainer="mhamidawad"
LABEL description="Qwen 0.6B Model with LoRA Adapter optimized for vLLM"
LABEL version="1.0"

# Set environment variables
ENV MODEL_NAME="Qwen/Qwen3-0.6B" \
    VLLM_PORT=8000 \
    VLLM_HOST="0.0.0.0" \
    PYTHONUNBUFFERED=1

# Copy model files (moved to repo root)
COPY adapter_config.json /model/adapter_config.json
COPY adapter_model.safetensors /model/adapter_model.safetensors
COPY chat_template.jinja /model/chat_template.jinja
COPY tokenizer.json /model/tokenizer.json
COPY tokenizer_config.json /model/tokenizer_config.json
COPY checkpoint-100/ /model/checkpoint-100/

# Install additional useful packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create health check script
RUN echo '#!/bin/bash\ncurl -s http://localhost:8000/health || exit 1' > /healthcheck.sh && \
    chmod +x /healthcheck.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /healthcheck.sh

# Expose port
EXPOSE 8000

# Optimized startup command with LoRA adapter
# GPU memory utilization, context length, and other parameters tuned for 0.6B model
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--model", "${MODEL_NAME}", \
     "--host", "${VLLM_HOST}", \
     "--port", "${VLLM_PORT}", \
     "--gpu-memory-utilization", "0.8", \
     "--max-model-len", "8192", \
     "--dtype", "auto", \
     "--enable-lora", \
     "--lora-modules", "my-qwen-adapter=/model", \
     "--max-lora-rank", "16", \
     "--lora-extra-vocab-size", "256", \
     "--tensor-parallel-size", "1", \
     "--pipeline-parallel-size", "1", \
     "--worker-use-ray", "false"]
