src_dir=$1
ResultsDir=$2

timeStamp=$(echo -n $(date "+%Y-%m-%d %H:%M:%S") | shasum | cut -f 1 -d " ")

mkdir -p ${ResultsDir}
results=${ResultsDir}/${timeStamp}

mkdir -p ${results}

exec 3>&1 4>&2
trap $(exec 2>&4 1>&3) 0 1 2 3
exec 1>${results}/${timeStamp}.log 2>&1

echo results at ${results}
echo STARTING at $(date)
git rev-parse HEAD

rules=("add_try_except_inside_functions")
# rules=("add_nested_for" "add_else_to_for" "add_nested_while" "add_nested_if" "changeToElif" "create_functions" "add_if_inside_functions" "add_try_except_inside_functions" "add_try_except_basic_code" "move_out_of_if_main" "replace_noreturn_func" "changing_AugAssign" "add_imports")

run_rules(){
    for rule in ${rules[@]}; do
        target_dir=${results}/target_${rule}
        mkdir -p ${target_dir}
        json=${results}/res_${rule}.json
        csv=${results}/res_${rule}.csv
        log=${results}/${rule}.log
        bash -x run_rules.sh ${src_dir} ${target_dir} ${json} ${csv} ${rule} |&tee ${log}
    done
}

run_combination(){
    target_dir=${results}/target_combination
    mkdir -p ${target_dir}
    json=${results}/res_combination.json
    csv=${results}/res_combination.csv
    log=${results}/combination.log
    python3 init.py -sd ${src_dir} -td ${target_dir} -j ${json} -c ${csv} |& tee ${log}

}

# run_combination
run_rules

echo ENDING at $(date)