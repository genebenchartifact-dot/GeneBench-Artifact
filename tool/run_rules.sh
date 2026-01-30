src_dir=$1
target_dir=$2
json=$3
csv=$4
rule=$5

echo STARTING at $(date)
git rev-parse HEAD

python3 init.py -sd ${src_dir} -td ${target_dir} -j ${json} -c ${csv} -r ${rule}

echo ENDING at $(date)