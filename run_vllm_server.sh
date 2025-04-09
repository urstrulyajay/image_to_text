#!/bin/bash

# Set variables
MODEL_DIR="merged_model"
PORT=8000
QUANTIZATION="none"  # or "awq", "gptq", etc if you quantized

# Check if vLLM is installed
if ! command -v python3 -m vllm.entrypoints.openai.api_server &> /dev/null
then
    echo "vLLM is not installed. Installing it now..."
    pip install vllm
fi

# Start the vLLM OpenAI-compatible server
echo "Launching vLLM server with model at $MODEL_DIR..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_DIR \
    --port $PORT \
    --quantization $QUANTIZATION \
    --trust-remote-code
