#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --job-name=train0
#SBATCH --account=bdsz-delta-gpu
#SBATCH --mem=64G
#SBATCH --nodes=1

echo STARTING at $(date)
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"

git rev-parse HEAD

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# accelerate launch --multi_gpu \
#   --num_processes=8 \
#   --mixed_precision bf16 \
#   qlora.py \
#   --model_name_or_path deepseek-ai/deepseek-coder-6.7b-base \
#   --dataset=data/train_all_shuffled.jsonl \
#   --no_gradient_checkpointing \
#   --num_train_epochs 1 \
#   --bf16 \
#   --output_dir full_output_shuffle_deepseek \
#   |& tee d_shuffle_dp.log

accelerate launch --multi_gpu \
  --num_processes=8 \
  --mixed_precision bf16 \
  qlora.py \
  --model_name_or_path codellama/CodeLlama-13b-Instruct-hf \
  --dataset=data/train_all_shuffled.jsonl \
  --no_gradient_checkpointing \
  --num_train_epochs 1 \
  --bf16 \
  --output_dir full_output_shuffle_codellama \
  |& tee d_shuffle_cl.log

  #codellama/CodeLlama-13b-Instruct-hf


echo END at $(date)
