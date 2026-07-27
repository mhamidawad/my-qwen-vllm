FROM vllm/vllm-openai-cpu:latest-x86_64

# Copy your Kaggle-trained adapter into the container
COPY my-qwen-model /my-qwen-model

# Bake all your hardware limits and model configs into the start command
CMD ["Qwen/Qwen3-0.6B", "--host", "0.0.0.0", "--port", "8000", "--gpu-memory-utilization", "0.5", "--max-model-len", "8192", "--dtype", "float32", "--enable-lora", "--lora-modules", "my-custom-model=/my-qwen-model"]
