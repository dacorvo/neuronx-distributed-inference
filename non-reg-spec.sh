#!/bin/bash
inference_demo  --model-type llama --task-type causal-lm export \
                --model-path ./Llama-3.1-8B-Instruct/ \
                --compiled-model-path Llama-3.1-8B-Instruct-traced \
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size 1 \
                --max-context-length 3892 \
                --seq-len 4096 \
                --speculation-length 5 \
                --on-device-sampling \
                --enable-bucketing \
                --pad-token-id 128001
inference_demo  --model-type llama --task-type causal-lm export \
                --model-path ./Llama-3.2-1B-Instruct/ \
                --compiled-model-path Llama-3.2-1B-Instruct-draft-traced \
                --torch-dtype bfloat16 \
                --tp-degree 8 \
                --batch-size 1 \
                --max-context-length 3892 \
                --seq-len 4096 \
                --on-device-sampling \
                --enable-bucketing \
                --pad-token-id 128001
# Generated outputs:
# Output 0: The color of the sky is blue because of a phenomenon called Rayleigh scattering,
# named after the British physicist Lord Rayleigh, who first described it in the late 19th century.
# Rayleigh scattering is the scattering of light by small particles or molecules in the atmosphere,
# such as nitrogen and oxygen molecules. When sunlight enters the Earth's atmosphere, it encounters
inference_demo  --model-type llama --task-type causal-lm run \
                --model-path Llama-3.1-8B-Instruct-traced \
                --draft-model-path Llama-3.2-1B-Instruct-draft-traced \
                --max-new-tokens 64 \
                --top-k 1 \
                --do-sample \
                --pad-token-id 128001 \
                --prompt "The color of the sky is"
