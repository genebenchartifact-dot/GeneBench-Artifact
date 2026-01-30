InputDir=$1
ProgramIDCSV=$2 # codenet_id.csv, avatar_id.csv
dataset=$3
# TranslationOutPutDir=$4
# Model=$5

MainDir=/home/yang/Benchmark
TestsDir=${MainDir}/dataset/python_avatar_tests

timeStamp=$(echo -n $(date "+%Y-%m-%d %H:%M:%S") | shasum | cut -f 1 -d " ")
OutputDir=${MainDir}/v3_metric_${dataset}_transformation/${timeStamp}

rules=(add_else_to_while
add_else_to_for
change_var_names
changing_AugAssign
add_nested_list
add_nested_for_in
change_function_names 
introduce_confusing_vars
move_functions_into_new_class
transform_range_to_recursion
add_try_except_inside_functions
add_nested_for_out 
add_nested_while_out 
add_nested_while_in 
add_nested_if 
create_functions 
replace_with_numpy  
add_thread
add_decorator)

echo STARTING at $(date)
git rev-parse HEAD
mkdir -p ${OutputDir}
mkdir -p ${OutputDir}/CSV

for rule in ${rules[@]}; do
    RuleResultBefore=${OutputDir}/CSV/${rule}_before.csv
    RuleResultAfter=${OutputDir}/CSV/${rule}_after.csv
    SimilarityResult=${OutputDir}/CSV/${rule}_similarity.csv
    echo "File, Base complexity, Predicates with operators, Nested levels, Variable max distance(sum), Variable max distance(average), Complex code structures, Third-Party calls, Inter_dependencies, Intra_dependencies, Final value, Log"> ${RuleResultBefore}
    echo "File, Base complexity, Predicates with operators, Nested levels, Variable max distance(sum), Variable max distance(average), Complex code structures, Third-Party calls, Inter_dependencies, Intra_dependencies, Final value, Log"> ${RuleResultAfter}
    echo "File1, File2, BLEU, CodeBLEU, Jaccard, LD, token_LD, token_distance_similarity" > ${SimilarityResult}
    cd ${MainDir}
    ProgramOutputDir=${OutputDir}/${dataset}/transformed/${rule}
    ResultOutputDir=${OutputDir}/${dataset}/transformed_results/${rule}_result
    mkdir -p ${ResultOutputDir}
    mkdir -p ${ProgramOutputDir}
    for program in $(cat $ProgramIDCSV); do
        python3 tool/init.py -s ${InputDir}/${program}.py -t ${ProgramOutputDir}/${program}.py \
        -j ${ResultOutputDir}/${program}.json -c ${ResultOutputDir}/${program}.csv -r ${rule} \
        # -e ${TestsDir}
        if [ -e ${ProgramOutputDir}/${program}.py ]
        then
            python3 tool/metrics/my_metric.py ${InputDir}/${program}.py ${ResultOutputDir} ${RuleResultBefore}
            python3 tool/metrics/my_metric.py ${ProgramOutputDir}/${program}.py ${ResultOutputDir} ${RuleResultAfter}
            python3 tool/metrics/similarity.py ${InputDir}/${program}.py ${ProgramOutputDir}/${program}.py ${SimilarityResult}
        fi
    done

    # cd /home/yang/PLTranslationEmpirical/dataset/${dataset}/Python/Code
    # rm -rf s*.py
    # cd ${MainDir}
    # cp -r ${ProgramOutputDir}/*.py /home/yang/PLTranslationEmpirical/dataset/${dataset}/Python/Code
    # cd /home/yang/PLTranslationEmpirical/
    # bash -x run.sh ${dataset} ${Model} ${TranslationOutPutDir}/transformation_${rule} 3 |& tee ${TranslationOutPutDir}/transformation_${rule}/translation_avatar_${rule}_log

done

mkdir -p ${OutputDir}/summary
python3 tool/metrics/combine_complexity.py ${OutputDir}/CSV/ ${OutputDir}/summary
python3 tool/metrics/combine_similarity.py ${OutputDir}/CSV/ ${OutputDir}/summary
for rule in ${rules[@]}; do
    python3 tool/metrics/correlation.py ${OutputDir}/summary/${rule}.csv ${OutputDir}/CSV//${rule}_similarity.csv
done
echo ENDING at $(date)