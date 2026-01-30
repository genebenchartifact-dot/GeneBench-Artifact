#!/bin/bash

set -e

MODELS=(
  # "deepseek-ai/deepseek-coder-33b-instruct"
  #   "codellama/CodeLlama-13b-Instruct-hf"
  #   "codellama/CodeLlama-13b-hf"
  #   "codellama/CodeLlama-34b-Instruct-hf"
  #   "WizardLM/WizardCoder-15B-V1.0"
  #   "semcoder/semcoder"
  #   "bigcode/starcoder2-15b"
  #   "deepseek-ai/deepseek-coder-6.7b-instruct"
  #   "deepseek-ai/deepseek-coder-6.7b-base"
)
CODE_DIR="dataset"
BUGS_DIR="program_repair/bugs_input" #"bugs_input"
OUTPUT_DIR="program_repair/Result_test_mul_codellama"
SCRIPT="program_repair/src/prompt_apr_pro.py"
SKIP=()


mkdir -p "$OUTPUT_DIR"

for MODEL in "${MODELS[@]}"; do
  for BUG_FILE in ${BUGS_DIR}/*.json; do
    BASENAME=$(basename "$BUG_FILE" .json)
    # skip
    if [[ " ${SKIP[*]} " =~ " ${BASENAME}.json " ]]; then
      echo "[INFO] Skipping ${BASENAME}.json"
      continue
    fi
    echo "[INFO] Processing $BASENAME with model $MODEL"

    if [[ "$BASENAME" == *"humaneval"* ]]; then
      DATASET="HumanEval"
      TESTS_DIR="$CODE_DIR/HumanEval/"
    elif [[ "$BASENAME" == *"classeval"* ]]; then
      DATASET="ClassEval"
      TESTS_DIR="$CODE_DIR/ClassEval/"
    else
      echo "[WARN] Skipping unknown dataset: $BASENAME"
      continue
    fi

    MODEL_TAG=$(echo "$MODEL" | sed 's/\//__/g')
    OUTFILE="${OUTPUT_DIR}/${BASENAME}_Result_${MODEL_TAG}.json"

    echo "Running: $SCRIPT --tests-dir ${TESTS_DIR} --model-name \"$MODEL\" --output-path $OUTFILE --bug-info-path $BUG_FILE --datasetname $DATASET"
    python3 "$SCRIPT" \
      --tests-dir "$TESTS_DIR" \
      --model-name "$MODEL" \
      --output-path "$OUTFILE" \
      --bug-info-path "$BUG_FILE" \
      --datasetname "$DATASET"

    echo "[DONE] Wrote: $OUTFILE"
    echo "----------------------------------------"
  done
done
