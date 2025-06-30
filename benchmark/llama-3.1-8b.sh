#!/bin/sh

model_id="unsloth/Llama-3.1-8b-instruct"
model_name="Llama-3.1-8b"
if [ -d "$model_name" ]; then
    echo "Model directory $model_name already exists. Skipping download."
else
    echo "Downloading model $model_id to $model_name..."
    huggingface-cli download ${model_id} --local-dir ${model_name}
fi
echo "Running benchmark for $model_name..."
for batch_size in 1 4 8 16 32 48; do
    neuron_model_name="${model_name}_bs${batch_size}_traced"
    if [ -d "$neuron_model_name" ]; then
        echo "Neuron model directory $neuron_model_name already exists. Skipping export."
    else
        echo "Exporting model with batch size $batch_size"
        inference_demo --model-type llama \
                    --task-type causal-lm \
                    run \
                    --model-path ${model_name} \
                    --compiled-model-path ${neuron_model_name} \
                    --torch-dtype bfloat16 \
                    --tp-degree 8 \
                    --batch-size ${batch_size} \
                    --is-continuous-batching \
                    --ctx-batch-size 1 \
                    --tkg-batch-size ${batch_size} \
                    --max-batch-size ${batch_size} \
                    --max-context-length 4096 \
                    --seq-len 4096 \
                    --on-device-sampling \
                    --pad-token-id 128001 \
                    --prompt "What is the meaning of life?" \
                    --prompt "What is the color of the sky?"
    fi
    echo "Running benchmark for batch size $batch_size..."
    python3 benchmark.py --model ${model_name} \
                         --neuron-model ${neuron_model_name} \
                         --inc-length 256 \
                         --max-length 2048 \
                         --name ${model_name}-bs${batch_size}
done
