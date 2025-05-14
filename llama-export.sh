#!/bin/bash
set -e
batch_size=${1:-8}
sequence_length=${2:-4096}
max_context_length=${3:-3892}
compiled_model_path=${4:-./traced/Llama-3.1-8B-Instruct-bs-${batch_size}-seq-${sequence_length}-max-${max_context_length}/}
inference_demo  --model-type llama \
                --task-type causal-lm \
                run \
                --model-path ./Llama-3.1-8B-Instruct/ \
                --compiled-model-path ${compiled_model_path} \
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size ${batch_size} \
                --max-context-length ${max_context_length} \
                --seq-len ${sequence_length} \
                --on-device-sampling \
                --enable-bucketing \
                --pad-token-id 128001 \
                --prompt "The color of the sky is"
# Generated outputs:
# Output 0: I believe the meaning of life is to find happiness and fulfillment in the present moment.
# It's a simple yet profound concept that can bring joy and peace to our lives.
# As I reflect on my own life, I realize that I've been focusing on the past and dwelling on what's to come.
# I've been trying to achieve certain goals and milestones,
# Output 1: The color of the sky is blue, but the color of the sky is not blue.
# This is a classic example of a logical fallacy called "appeal to appearance" or "appeal to taste."
# It is a mistake to assume that the color of the sky is blue because it appears blue to us.
# The color of the sky is actually
