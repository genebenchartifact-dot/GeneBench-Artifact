#!/bin/bash

MainDir=$(pwd)
DatasetList=("cruxeval" "HumanEval" "ClassEval" "Avatar") #"cruxeval" "ClassEval" "Avatar" "HumanEval"
MAX_SESSIONS=$1   # limit of concurrent tmux sessions
GADIR="GA_v1"
echo STARTING at $(date)
git rev-parse HEAD
mkdir -p "$GADIR"
mkdir -p ${GADIR}_outputs

source venv/bin/activate
python -c "import nltk; nltk.download('punkt', quiet=True)"
python -c "import nltk; nltk.download('punkt_tab', quiet=True)"
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', quiet=True)"
#nltk.download('averaged_perceptron_tagger_eng')
pip install autopep8

for dataset in "${DatasetList[@]}"; do
  if [ "$dataset" = "Avatar" ]; then
    InputCodeDir="${MainDir}/dataset/${dataset}/Code/"
  else
    InputCodeDir="${MainDir}/dataset/${dataset}/code/"
  fi
  OutputCodeDir="${MainDir}/${GADIR}_outputs/${dataset}"
  EvaluationDir="${MainDir}/dataset/${dataset}"
  

  mkdir -p "${OutputCodeDir}"

  # Get program IDs
  # if [ "$dataset" = "cruxeval" ] || [ "$dataset" = "Avatar" ]; then
  IdFile="${EvaluationDir}/ids.txt"
  ids=$(cat "$IdFile")
  # elif [ "$dataset" = "HumanEval" ]; then
  #   ids=$(seq 0 163)
  # elif [ "$dataset" = "ClassEval" ]; then
  #   ids=$(seq 0 99)
  # fi

  for i in $ids; do
    # if [ "$dataset" = "cruxeval" ] || [ "$dataset" = "Avatar" ]; then
    program="$i"
    # else
    #   program="${dataset}_${i}"
    # fi

    source_file="${InputCodeDir}/${program}.py"
    if [ ! -f "$source_file" ]; then
      echo "Source file not found: $source_file"
      continue
    fi

    timeStamp=$(echo -n $(date "+%Y-%m-%d %H:%M:%S") | shasum | cut -f 1 -d " ")
    ResultDir="${GADIR}/${dataset}/${program}/${timeStamp}"
    mkdir -p "${ResultDir}"

    # Wait if too many tmux sessions are active
    while [ "$(tmux ls 2>/dev/null | wc -l)" -ge "$MAX_SESSIONS" ]; do
      echo "Reached $MAX_SESSIONS tmux sessions. Waiting..."
      sleep 10
    done

    echo "Launching tmux session for $program..."
    tmux new-session -d -s "$program" bash -c "
      source venv/bin/activate
      python3 -u tool/init.py \
        --source_file \"$source_file\" \
        --target_file \"${OutputCodeDir}/${program}_transformed.py\" \
        --file_id \"$program\" \
        --evaluation_tests_dir \"$EvaluationDir\" \
        --genetic_algorithm \
        --result_dir \"$ResultDir\" --time_budget 3600 |& tee \"$ResultDir/${dataset}.log\"

      echo \"Completed $program at \$(date)\"
    "
  done
done

echo ALL TMUX JOBS LAUNCHED at $(date)
