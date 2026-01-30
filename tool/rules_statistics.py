import os
import sys
import csv
import utils

def main(dir, save_file):
    fields = {'rules': None,'num':None, 'effort_before':None, 'effort_after':None, 'difficulty_before':None, 'difficulty_after':None,
        'calculated_length_before':None, 'calculated_length_after':None, 'length_before':None, 'length_after':None, 'vocabulary_before':None, 'vocabulary_after':None,
        'mi_value_before':None, 'mi_value_after':None, 'Nloc_before':None, 'Nloc_after':None, 'AvgCCN_before':None, 'AvgCCN_after':None, 'codebleu': None}

    utils.write_header_csv(save_file,[key for key in fields])
    for dirpath, _, files in os.walk(dir):
        for file in files:
            # print(file)
            fields = {'rules': None,'num':None, 'effort_before':None, 'effort_after':None, 'difficulty_before':None, 'difficulty_after':None,
            'calculated_length_before':None, 'calculated_length_after':None, 'length_before':None, 'length_after':None, 'vocabulary_before':None, 'vocabulary_after':None,
            'mi_value_before':None, 'mi_value_after':None, 'Nloc_before':None, 'Nloc_after':None, 'AvgCCN_before':None, 'AvgCCN_after':None, 'codebleu': None}
            if file.endswith('.log'):
                file_path = os.path.join(dirpath, file)
                rule = file.split('.')[0]
                content = utils.read_file(file_path)
                fields['rules'] = rule
                for line in content.split('\n'):
                    if 'Rules applied for:' in line:
                        num = line.split(':')[-1].strip()
                        fields['num'] = num
                    for key in fields:
                        if key == 'num' or key == 'rules':
                            continue
                        if key in line:
                            fields[key] = line.split(' ')[-1].strip()
                print(rule, fields)
                utils.write_dict_csv(save_file,[key for key in fields],fields)
                # print("1")

if __name__ == "__main__":
    args = sys.argv[1:]
    dir_after = args[0]
    # saveToFile = args[1]
    saveToFile = dir_after.split("/")[-1] + ".csv"
    # main('/home/yang/contamination/tool/avatar_python_create_func/', save_file)
    main(dir_after, saveToFile)