INPUTDIR=GA1/HumanEval/
DATASET=HumanEval
OUTPUT=renamed/${INPUTDIR}

mkdir -p ${OUTPUT}

python tool/gemini_rename.py \
  --file_dir $INPUTDIR \
  --dataset $DATASET \
  --tests_dir dataset/$DATASET \
  --new_dir ${OUTPUT} |& tee ${OUTPUT}/rename_${DATASET}.log
