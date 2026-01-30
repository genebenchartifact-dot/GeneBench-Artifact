#!/bin/bash

models=(
    "codellama/CodeLlama-13b-hf"
    "deepseek-ai/deepseek-coder-6.7b-base"
    "deepseek-ai/deepseek-coder-6.7b-instruct"
    "deepseek-ai/deepseek-coder-33b-instruct"
    "codellama/CodeLlama-13b-Instruct-hf"
    "codellama/CodeLlama-34b-Instruct-hf"
    "WizardLM/WizardCoder-15B-V1.0"
    "bigcode/starcoder2-15b"
    "semcoder/semcoder"
)

temperatures=(0.01)

for ((i=0; i<${#models[@]}; i++)); do
    model=${models[$i]}
    base_dir=${models[$i]}
    echo $model
    for temperature in "${temperatures[@]}"; do
        dir="${base_dir}_temp${temperature}_input"

dir=$dir
SIZE=800
GPUS=2


echo ${dir}
mkdir -p model_generations_raw/${dir}

string="Starting iteration ${i} with start and end  \$((\$i*SIZE/GPUS)) \$((\$ip*SIZE/GPUS))"
echo \$string

python3 main.py \
    --model $model \
    --use_auth_token \
    --trust_remote_code \
    --tasks input_prediction \
    --batch_size 10 \
    --n_samples 1 \
    --max_length_generation 2048 \
    --precision bf16 \
    --temperature $temperature \
    --save_generations \
    --save_generations_path model_generations_raw/${dir}/shard_$((${i})).json \
    --cot \
    --shuffle \
    --tensor_parallel_size 2
    done
done