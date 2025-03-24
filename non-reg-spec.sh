#!/bin/bash
inference_demo  --model-type llama --task-type causal-lm run \
                --model-path ./Llama-3.1-8B-Instruct/ \
                --compiled-model-path Llama-3.1-8B-Instruct-traced \
                --draft-model-path ./Llama-3.2-1B-Instruct/ \
                --compiled-draft-model-path Llama-3.2-1B-Instruct-draft-traced \
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size 1 \
                --max-context-length 3892 \
                --seq-len 4096 \
                --speculation-length 5 \
                --no-trace-tokengen-model \
                --max-new-tokens 64 \
                --enable-bucketing \
                --top-k 1 \
                --do-sample \
                --pad-token-id 128001 \
                --prompt "The color of the sky is"
# Generated outputs:
# Output 0: The color of the sky is blue, but what about the color of the sky at night?
# Is it blue? No, it's not blue. The color of the sky at night is actually a deep shade of indigo or purple,
# depending on the time of year and atmospheric conditions.
#
#The color of the sky is determined by the way
