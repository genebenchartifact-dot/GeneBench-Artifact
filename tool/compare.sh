transformation_avatar=$1
transformation_codenet=$2
save_to_numbers_csv=$3
save_to_pass=$4

transformation_avatar="/home/yang/contamination/tool/latest_avatar.csv"
transformation_codenet="/home/yang/contamination/tool/latest_codenet_2.csv"
save_to_numbers_csv="numbers.csv"
save_to_pass="p"

python3 generate_data.py ${transformation_avatar} ${transformation_codenet} ${save_to_numbers_csv} ${save_to_pass}

python3 generate_pass_data.py ${save_to_pass} >> ${save_to_numbers_csv}