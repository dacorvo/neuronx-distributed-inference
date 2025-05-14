#!/bin/bash
set -e
batch_size=${1:-8}
sequence_length=${2:-4096}
max_context_length=${3:-3892}
compiled_model_path=${4:-./traced/Llama-3.1-8B-Instruct-bs-${batch_size}-seq-${sequence_length}-max-${max_context_length}/}
inference_demo  --model-type llama\
                --task-type causal-lm \
                run \
                --model-path ./Llama-3.1-8B-Instruct/ \
                --compiled-model-path ${compiled_model_path}
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size ${batch_size} \
                --max-context-length ${max_context_length} \
                --seq-len ${sequence_length} \
                --on-device-sampling \
                --enable-bucketing \
                --pad-token-id 128001 \
                --top-k 1 \
                --do-sample \
                --skip-compile \
                --prompt "The color of the sky is"
