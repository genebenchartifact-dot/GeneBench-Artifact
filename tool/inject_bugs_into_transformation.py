import os
import sys
import csv
import utils
import json

import ast
import astor


import difflib

def apply_bug_by_keyword(line: str, keyword: str) -> tuple[str, bool]:
    original = line
    line = line.strip()

    if keyword == "return" and line.startswith("return "):
        return "return not " + line[len("return "):], True

    if keyword == "if" and line.startswith("if "):
        return "if not " + line[len("if "):], True

    return original, False

def inject_bugs_by_common_keywords(code1: str, code2: str):
    keywords = {"return", "if"}

    lines1 = code1.strip().split('\n')
    lines2 = code2.strip().split('\n')

    def find_bug_opportunities(lines):
        keyword_lines = {kw: [] for kw in keywords}
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("return "):
                keyword_lines["return"].append(idx)
            if stripped.startswith("if "):
                keyword_lines["if"].append(idx)
        return keyword_lines

    kw_lines1 = find_bug_opportunities(lines1)
    kw_lines2 = find_bug_opportunities(lines2)

    # Determine common keywords
    common_keywords = {kw for kw in keywords if kw_lines1[kw] and kw_lines2[kw]}

    # Cap bug count per keyword to minimum opportunities
    bug_limits = {kw: min(len(kw_lines1[kw]), len(kw_lines2[kw])) for kw in common_keywords}

    def apply(lines, kw_lines, bug_limits):
        new_lines = lines.copy()
        bug_records = []
        for keyword, indices in kw_lines.items():
            count = 0
            for idx in indices:
                if count >= bug_limits.get(keyword, 0):
                    break
                buggy_line, is_buggy = apply_bug_by_keyword(lines[idx], keyword)
                if is_buggy:
                    indent = len(lines[idx]) - len(lines[idx].lstrip())
                    modified_line = " " * indent + buggy_line
                    new_lines[idx] = modified_line
                    bug_records.append((idx + 1, lines[idx], modified_line))
                    count += 1
        return "\n".join(new_lines), bug_records

    code1_mod, bugs1 = apply(lines1, kw_lines1, bug_limits)
    code2_mod, bugs2 = apply(lines2, kw_lines2, bug_limits)

    return (code1_mod, bugs1), (code2_mod, bugs2)

def get_python_files_dir(directory):
    file_list = []
    for dir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py") or ".py" in file:
                file_path = os.path.join(dir, file)
                file_list.append(file_path)
    return file_list

def get_dataset(dataset_dir):
    dataset = {}
    file_paths = get_python_files_dir(dataset_dir)
    for file_path in file_paths:
        file_ID = file_path.split("/")[-1].split(".py")[0]
        if file_ID not in dataset:
            dataset[file_ID] = {
                "file": file_path,
                "correct_code": utils.read_file(file_path)
            }
    return dataset

def get_datasets(dataset_dir):
    dataset = {}
    file_paths = get_python_files_dir(dataset_dir)
    for file_path in file_paths:
        if file_path.endswith(".csv") or file_path.endswith(".png"):
            continue
        file_ID = file_path.split("/")[-1].split(".py")[0]
        if file_ID not in dataset:
            dataset[file_ID] = {
                "file": []
            }
        dataset[file_ID]["file"].append(file_path)
    return dataset

def inject_buggy_stats(correct_code, target_file, bugs):
    tag = False
    num = 0
    print(target_file, "original bugs:", len(bugs))
    num = 0
    for original_stat in bugs:
        if original_stat in correct_code:
            # print('1')
            # exit(0)
            correct_code.replace(original_stat, bugs[original_stat])
            print(f"{target_file}: correct: {original_stat}, buggy: {bugs[original_stat]}")
            num += 1
            tag = True
    print(target_file, "new bugs:", num)
    if tag:
        utils.write_file(target_file, correct_code)
    return tag

def main(dataset, bug_info, backup):
    for ID in dataset:
        # print(ID)
        # exit(0)
        index = f"/home/yang/Benchmark/dataset/python_codenet/{ID}.py"
        if index not in bug_info:
            print(f"{ID} not in bug_info")
            continue
        file = dataset[ID]["file"]
        correct_code = dataset[ID]["correct_code"]
        bugs = bug_info[index]["diff"]
        
        name_seed = utils.generate_sha(file)
        tmp_dir = "./apr_bugs_transformation"
        target_file = f"{tmp_dir}/{file.split('/')[-1].split('-')[0]}-{name_seed}" 
        tag = inject_buggy_stats(correct_code, target_file, bugs)
        if not tag:
            for file_path in backup[ID]["file"]:
                correct_code = utils.read_file(file_path)
                name_seed = utils.generate_sha(file_path)
                tmp_dir = "./apr_bugs_transformation"
                target_file = f"{tmp_dir}/{file_path.split('/')[-1].split('-')[0]}-{name_seed}" 
                tag = inject_buggy_stats(correct_code, target_file, bugs)
                if tag:
                    break
                
        
def read_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def inject_bugs_per_case(program_file, bugs, tmp_dir):
    correct_code = utils.read_file(program_file)
    applied = {"diff":{}}
    
    name_seed = utils.generate_sha(program_file)
    target_file = f"{tmp_dir}/{program_file.split('/')[-1].split('-')[0]}-{name_seed}"
    for original_stat in bugs["diff"]:
        if original_stat in correct_code:
            new_code = correct_code.replace(original_stat, bugs["diff"][original_stat])
            correct_code = new_code
            applied["diff"].update({original_stat: bugs["diff"][original_stat]})
    if len(applied["diff"]):
        utils.write_file(target_file, correct_code)
    else:
        print("=======================")
        # print(correct_code)
        # print("=======================")
        # exit(0)
    return applied, target_file
    
def inject_bugs(bug_info, original_dataset, transformation_programs):
    original_dir = "original_bugs_humaneval1"
    transformation_dir = "transformation_bugs_humaneval1"
    for original_file_ID in bug_info:
        ID = original_file_ID.split("/")[-1].split(".py")[0]
        print(ID)
        if ID not in transformation_programs:
            continue
        
        transformation_file = transformation_programs[ID]["file"][0]
        original_file = original_dataset[ID]["file"][0]

        print(transformation_file, original_file)
        # exit(0)
        applied, transformation_bugs_file = inject_bugs_per_case(transformation_file, bug_info[original_file_ID], transformation_dir)
        applied, original_bugs_file = inject_bugs_per_case(original_file, applied, original_dir)
        if len(applied['diff']) > 0:
            print(f"Applied bugs: {ID}")
            print(f"{applied},{original_bugs_file}, {transformation_bugs_file}")
        else:
            code1 = utils.read_file(original_file)
            code2 = utils.read_file(transformation_file)
            (out1, bugs1), (out2, bugs2) = inject_bugs_by_common_keywords(code1, code2)
            if len(bugs1) and len(bugs2):
                utils.write_file(original_bugs_file, out1)
                utils.write_file(transformation_bugs_file, out2)
            # print("Transformed Code 1:\n", out1)
            # print("Bugs in Code 1:", bugs1)
            # print("\nTransformed Code 2:\n", out2)
            # print("Bugs in Code 2:", bugs2)
            # exit(0)
        
if __name__ == "__main__":
    args = sys.argv[1:]
    original_programs = args[0]
    transformations = args[1]
    bugs = args[2] # humaneval_bugs.json; codenet_bugs.json
    
    bug_info = read_json(bugs)
    original_dataset = get_datasets(original_programs)
    transformation_programs = get_datasets(transformations)
    inject_bugs(bug_info, original_dataset, transformation_programs)