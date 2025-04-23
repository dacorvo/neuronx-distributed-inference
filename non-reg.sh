#!/bin/bash
set -e
inference_demo  --model-type llama --task-type causal-lm export \
                --model-path ./Llama-3.2-1B-Instruct/ \
                --compiled-model-path Llama-3.2-1B-Instruct-traced \
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size 4 \
                --max-context-length 3892 \
                --seq-len 4096 \
                --on-device-sampling \
                --enable-bucketing \
                --pad-token-id 128001
# Generated outputs:
# Output 0: I believe the meaning of life is to find happiness and fulfillment in the present moment.
# It's a simple yet profound concept that can bring joy and peace to our lives.
# As I reflect on my own life, I realize that I've been focusing on the past and dwelling on what's to come.
# I've been trying to achieve certain goals and milestones,
# Output 1: The color of the sky is blue, but the color of the sky is not blue.
# This is a classic example of a logical fallacy called "appeal to appearance" or "appeal to taste."
# It is a mistake to assume that the color of the sky is blue because it appears blue to us.
# The color of the sky is actually
inference_demo  --model-type llama --task-type causal-lm run \
                --model-path ./Llama-3.2-1B-Instruct-traced/ \
                --max-new-tokens 64 \
                --top-k 1 \
                --do-sample \
                --pad-token-id 128001 \
                --prompt "I believe the meaning of life is" \
                --prompt "The color of the sky is"
