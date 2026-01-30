InputDir=$1
ProgramIDCSV=$2 # codenet_id.csv, avatar_id.csv
dataset=$3
# TranslationOutPutDir=$4
# Model=$5

MainDir=/home/yang/Benchmark
TestsDir=${MainDir}/dataset/python_avatar_tests

timeStamp=$(echo -n $(date "+%Y-%m-%d %H:%M:%S") | shasum | cut -f 1 -d " ")
OutputDir=${MainDir}/v6_metric_${dataset}_transformation_ga/${timeStamp}

echo STARTING at $(date)
git rev-parse HEAD
mkdir -p ${OutputDir}
mkdir -p ${OutputDir}/CSV

rule=ga

RuleResultBefore=${OutputDir}/CSV/${rule}_before.csv
RuleResultAfter=${OutputDir}/CSV/${rule}_after.csv
SimilarityResult=${OutputDir}/CSV/${rule}_similarity.csv
echo "File, Base complexity, Predicates with operators, Nested levels, Variable max distance(sum), Variable max distance(average), Complex code structures, Third-Party calls, Inter_dependencies, Intra_dependencies, Final value, Log"> ${RuleResultBefore}
echo "File, Base complexity, Predicates with operators, Nested levels, Variable max distance(sum), Variable max distance(average), Complex code structures, Third-Party calls, Inter_dependencies, Intra_dependencies, Final value, Log"> ${RuleResultAfter}
# echo "File1, File2, BLEU, CodeBLEU, Jaccard, LD, token_LD, token_distance_similarity" > ${SimilarityResult}
cd ${MainDir}
ProgramOutputDir=${OutputDir}/${dataset}/transformed/${rule}
ResultOutputDir=${OutputDir}/${dataset}/transformed_results/${rule}_result
mkdir -p ${ResultOutputDir}
mkdir -p ${ProgramOutputDir}
for program in $(cat $ProgramIDCSV); do
    python3 -u tool/init.py -s ${InputDir}/${program}.py -t ${ProgramOutputDir}/${program}.py \
    -j ${ResultOutputDir}/${program}.json -c ${ResultOutputDir}/${program}.csv -ga genetic_algorithm |& tee ${ResultOutputDir}/${program}.log
    # -e ${TestsDir}
    if [ -e ${ProgramOutputDir}/${program}.py ]
    then
        python3 tool/metrics/my_metric.py ${InputDir}/${program}.py ${ResultOutputDir} ${RuleResultBefore}
        python3 tool/metrics/my_metric.py ${ProgramOutputDir}/${program}.py ${ResultOutputDir} ${RuleResultAfter}
        # python3 tool/metrics/similarity.py ${InputDir}/${program}.py ${ProgramOutputDir}/${program}.py ${SimilarityResult}
    fi
done

# mkdir -p ${OutputDir}/summary
# python3 tool/metrics/combine_complexity.py ${OutputDir}/CSV/ ${OutputDir}/summary
# python3 tool/metrics/combine_similarity.py ${OutputDir}/CSV/ ${OutputDir}/summary
echo ENDING at $(date)