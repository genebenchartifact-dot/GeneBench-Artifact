InputsDir=$1 #python_avatar, python_codenet, python_cruxeval, python_humaneval, python_mbpp
OutputsDir=$2
ProgramIDCSV=$3
Dataset=$4

timeStamp=$(echo -n $(date "+%Y-%m-%d %H:%M:%S") | shasum | cut -f 1 -d " ")
echo STARTING at $(date)
git rev-parse HEAD
echo ${timeStamp}

ResultDir=${OutputsDir}/${Dataset}/${timeStamp}
ResultCSVDir=${OutputsDir}/${Dataset}/${timeStamp}/CSV
Complexity=/home/yang/Benchmark/tool/metrics/my_metric.py
ResultCSV=${ResultCSVDir}/${Dataset}_complexity.csv

mkdir -p ${OutputsDir}
mkdir -p ${ResultDir}
mkdir -p ${ResultCSVDir}
echo "File, Base complexity, Predicates with operators, Nested levels, Complex code structures, Third-Party calls, Inter_dependencies, Intra_dependencies, Log" >${ResultCSV}

for program in $(cat $ProgramIDCSV); do
    timeout 60 python3 ${Complexity} ${InputsDir}/${program} ${ResultDir} ${ResultCSV} |& tee ${ResultDir}/${timeStamp}.log
done

echo END at $(date)

# Get python files/IDs
# python3 tool/metrics/my_metric.py ${InputDir}/${program}.py ${ResultOutputDir} >> ${RuleResultBefore}
