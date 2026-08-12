#!/usr/bin/env python3
"""
Example script for using Qwen LoRA model with vLLM OpenAI-compatible API

Make sure vLLM server is running first:
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen3-0.6B \
        --enable-lora \
        --lora-modules my-qwen-adapter=./
"""

import sys
from openai import OpenAI
import time


def example_simple_chat():
    """Simple chat completion example"""
    print("\n" + "="*60)
    print("Example 1: Simple Chat Completion")
    print("="*60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-unused"
    )
    
    response = client.chat.completions.create(
        model="my-qwen-adapter",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is machine learning in simple terms?"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=0.9
    )
    
    print(f"User: What is machine learning in simple terms?")
    print(f"Assistant: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")


def example_streaming():
    """Streaming response example"""
    print("\n" + "="*60)
    print("Example 2: Streaming Response")
    print("="*60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-unused"
    )
    
    print("User: Write a short poem about artificial intelligence")
    print("Assistant: ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="my-qwen-adapter",
        messages=[
            {"role": "user", "content": "Write a short poem about artificial intelligence"}
        ],
        max_tokens=200,
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()  # Newline after streaming


def example_batch_processing():
    """Batch processing multiple prompts"""
    print("\n" + "="*60)
    print("Example 3: Batch Processing Multiple Prompts")
    print("="*60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-unused"
    )
    
    prompts = [
        "Explain quantum computing",
        "What is blockchain?",
        "How do neural networks work?"
    ]
    
    print(f"Processing {len(prompts)} prompts...\n")
    
    start_time = time.time()
    
    for i, prompt in enumerate(prompts, 1):
        response = client.chat.completions.create(
            model="my-qwen-adapter",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        
        print(f"{i}. Q: {prompt}")
        print(f"   A: {response.choices[0].message.content[:200]}...")
        print()
    
    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Average time per prompt: {elapsed/len(prompts):.2f} seconds")


def example_list_models():
    """List available models and adapters"""
    print("\n" + "="*60)
    print("Example 4: List Available Models")
    print("="*60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-unused"
    )
    
    models = client.models.list()
    print("Available models/adapters:")
    for model in models.data:
        print(f"  - {model.id}")


def example_with_system_prompt():
    """Example with detailed system prompt"""
    print("\n" + "="*60)
    print("Example 5: Using System Prompt for Role-Playing")
    print("="*60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-unused"
    )
    
    system_prompt = """You are an expert Python programmer with 20 years of experience.
You help users by providing clean, efficient, and well-documented code.
You always explain your reasoning and suggest best practices."""
    
    response = client.chat.completions.create(
        model="my-qwen-adapter",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How do I efficiently read a large CSV file in Python?"}
        ],
        temperature=0.5,  # Lower temperature for more focused responses
        max_tokens=300
    )
    
    print("System: Expert Python Programmer")
    print("User: How do I efficiently read a large CSV file in Python?")
    print(f"Assistant:\n{response.choices[0].message.content}")


def example_temperature_comparison():
    """Show how temperature affects responses"""
    print("\n" + "="*60)
    print("Example 6: Temperature Comparison (Creativity)")
    print("="*60)
    
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="token-unused"
    )
    
    prompt = "What is the meaning of life?"
    temperatures = [0.1, 0.7, 1.5]
    
    print(f"Prompt: '{prompt}'")
    print()
    
    for temp in temperatures:
        response = client.chat.completions.create(
            model="my-qwen-adapter",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100
        )
        
        print(f"Temperature {temp} (lower=more consistent, higher=more creative):")
        print(f"  Response: {response.choices[0].message.content[:150]}...")
        print()


def main():
    """Run all examples"""
    try:
        # Test connection first
        client = OpenAI(
            base_url="http://localhost:8000/v1",
            api_key="token-unused"
        )
        client.models.list()
        print("✓ Connected to vLLM server successfully!")
        
    except Exception as e:
        print(f"✗ Failed to connect to vLLM server")
        print(f"  Error: {e}")
        print(f"\n  Make sure vLLM server is running:")
        print(f"    python -m vllm.entrypoints.openai.api_server \\")
        print(f"      --model Qwen/Qwen3-0.6B \\")
        print(f"      --enable-lora \\")
        print(f"      --lora-modules my-qwen-adapter=./")
        sys.exit(1)
    
    # Run examples
    try:
        example_simple_chat()
        example_streaming()
        example_list_models()
        example_batch_processing()
        example_with_system_prompt()
        example_temperature_comparison()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
