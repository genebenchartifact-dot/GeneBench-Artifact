import ast
import csv
# import editdistance
import json
import math
import os
import sys
import hashlib
import subprocess
import readability
# from codebleu import calc_codebleu
# from nltk import word_tokenize
import numpy as np

class TextColor:
    HEADER = '\033[95m'
    ENDC = '\033[0m'

readability_operators = { 
    'num_tokens': ["add_nested_for_out", "add_nested_while_out", "add_nested_if", "create_functions", "add_try_except_inside_functions", "add_else_to_for",
                "add_else_to_while", "add_nested_list", "replace_with_numpy", "transform_range_to_recursion", "add_thread", "add_decorator", "introduce_confusing_vars"], 
    'lines_of_code': ["add_nested_for_out", "add_nested_while_out", "add_nested_if", "create_functions", "add_try_except_inside_functions", "add_else_to_for",
                "add_else_to_while", "replace_with_numpy", "transform_range_to_recursion", "add_thread", "add_decorator", "introduce_confusing_vars"],
    'num_conditions': ["add_nested_if", "transform_range_to_recursion"],
    'num_loops': ["add_nested_for_out", "add_nested_while_out"],
    'num_assignments': ["add_nested_for_out", "add_nested_while_out", "add_nested_if", "transform_range_to_recursion", "add_thread", "add_decorator", "introduce_confusing_vars"],
    'num_max_nested_loop': ["add_nested_for_out", "add_nested_while_out"], 
    'num_max_nested_if': ["add_nested_if"], 
    'num_max_conditions_in_if': ["add_nested_if"],
    'max_line_tokens': [], 
    'num_of_variables': ["introduce_confusing_vars"], 
    'num_of_arrays': ["add_nested_list"], 
    'num_of_operators': ["add_nested_for_out", "add_nested_while_out", "add_nested_if"], 
    # 'num_of_missing_conditions': [], 
    'num_of_nested_casting': [], 
    'entropy': ["add_nested_for_out", "add_nested_while_out", "add_nested_if", "create_functions", "add_try_except_inside_functions", "add_else_to_for",
                "add_else_to_while", "add_nested_list", "replace_with_numpy", "transform_range_to_recursion", "add_thread", "add_decorator", "introduce_confusing_vars"]
}


def dump_json(file_path, data):
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_sets(i) for i in obj]
        return obj

    # Transform the data and write to file
    data = convert_sets(data)
    
    with open(file_path, 'w') as fp:
        json.dump(data, fp)


def generate_sha(input_string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_string.encode('utf-8'))
    return sha256_hash.hexdigest()

def readability_stopping(readability, target):
    for key in readability:
        if readability[key] > target[key]:
            return readability_operators[key], {key:readability[key]}
    return [], None

def color_print_line(line):
    return f"{TextColor.HEADER}{line}{TextColor.ENDC}"

def tokenize_code(code):
    tokens = word_tokenize(code)
    return tokens

def clear_string(old_string):
    lines = old_string.splitlines()
    stripped_lines = [line.rstrip() for line in lines]
    cleaned_string = "\n".join(stripped_lines)
    return cleaned_string

def write_header_csv(csv_path, fields):
    """To set up headers for a csv file. 
       Followed by the following function `write_dict_csv` to write content in the csv file.
    """
    print(csv_path)
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

def write_dict_csv(csv_path, fields, dict_data):
    """To add rows into a csv file with property `a`.
       Usually after setting up csv headers by calling the above function `write_header_csv`.
    """
    with open(csv_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerow(dict_data)

def create_import(name, asname):
    """Create and return an ast Import node in the following format:
    import name as asname
    """
    ImportNode = ast.Import(names=[ast.alias(name=name, asname=asname)])
    return ImportNode

def create_importFrom(module, name, asname, level):
    """Create and return an ast ImportFrom node in the following format:
    from module import name as asname
    level=0 for absolute import, level=1 for relative import (e.g., .module).
    """
    ImportFromNode = ast.ImportFrom(
        module= module,
        names=[ast.alias(name=name, asname=asname)],
        level=level
    )
    return ImportFromNode

def get_imports(root):
    import_nodes = []
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.Import):
            if node not in import_nodes:
                import_nodes.append(node)
        elif isinstance(node, ast.ImportFrom):
            if node not in import_nodes:
                import_nodes.append(node)
    return import_nodes

def read_file(file_path):
    file = open(file_path, 'r')
    content = file.read()
    return content

def write_file(file_path,content):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except:
        pass
    f = open(file_path, "w")
    f.write(content)
    f.close()

def cal_codebleu(content1, content2):
    score = calc_codebleu([content1], [content2], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
    return score

def write_json(file_path, dict):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    except:
        pass
    with open(file_path, 'w') as fp:
        json.dump(dict, fp, indent=4)

def git_diff(file_path):
    diff_path = file_path.replace(".py","_diff.patch")
    diff_opts = ["git", "diff", file_path] #, "|& tee", diff_path
    print(" ".join(diff_opts), flush=True)
    diff = subprocess.run(diff_opts, stdout=subprocess.PIPE)
    diff_output = diff.stdout.decode('utf-8')
    write_file(diff_path, diff_output)
    # print(diff_output)
    return diff_path

def diff(source_file, target_file):
    diff_path = target_file.replace(".py","_diff.patch")
    diff_opts = ["diff", "-u", source_file, target_file]
    diff_output = run_cmds(diff_opts, None)
    write_file(diff_path, diff_output)
    return diff_output, diff_path

def read_json(file_path):
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    return data

def get_complexity(source_file, result_dir):
    print(f"arguments: {source_file}, {result_dir}")
    # try:
    if True:
        complexity_options = ["python3", "tool/metrics/my_metric.py", source_file, result_dir, "tmp.csv"]
        metrics = run_cmds(complexity_options, None)
        print(metrics)
        return ast.literal_eval(metrics)
    # except Exception as e:
    #     print(e)
    #     return None

def get_similarity(file1, file2, result_csv):
    similarity_options = ["python3", "tool/metrics/similarity.py", file1, file2, result_csv]
    similarity = run_cmds(similarity_options, None)
    return similarity

# def token_LD_distance(file1_tokens, file2_tokens):
#     token_LD = editdistance.eval(file1_tokens, file2_tokens)
#     return token_LD

def get_similarity_between_files(file1, file2):
    file1_tokens = tokenize_code(read_file(file1))
    file2_tokens = tokenize_code(read_file(file2))
    token_ld_distance = token_LD_distance(file1_tokens, file2_tokens)
    return token_ld_distance

def get_readability(file):
    return readability.get_readability_for_file(file)

def run_cmds(cmd_list, timeoutVal):
    """To run commands with subprocess and set up a timeout value, stdout is `subprocess.PIPE`
        Command will be printed, output be returned.
    """
    cmds = " ".join(cmd_list)
    # print("Generating with `{}`...".format(cmds))
    if timeoutVal != None:
        run_cmds = subprocess.run(cmd_list, stdout=subprocess.PIPE, timeout=timeoutVal) #check=True, capture_output=True, shell=True
    else:
        run_cmds = subprocess.run(cmd_list, stdout=subprocess.PIPE)
    output = run_cmds.stdout.decode('utf-8')
    return output

def run_cmds_nopipe(cmd_list, timeoutVal):
    """To run commands with subprocess but without PIPE and set up a timeout value.
        Command will be printed, output be returned.
    """
    cmds = " ".join(cmd_list)
    print(cmds, flush=True)
    if timeoutVal != None:
        run_cmds = subprocess.run(cmds, check=True, capture_output=True, shell=True, timeout=timeoutVal) #check=True, capture_output=True, shell=True
    else:
        run_cmds = subprocess.run(cmds, check=True, capture_output=True, shell=True)
    output = run_cmds.stdout.decode('utf-8')
    # print(output, flush=True)
    return output

def git_checkout(file_path):
    git_checkout_opts = ["git", "checkout", file_path]
    print(" ".join(git_checkout_opts), flush=True)
    git_checkout = subprocess.run(git_checkout_opts, stdout=subprocess.PIPE)
    git_checkout_output = git_checkout.stdout.decode('utf-8')
    return git_checkout_output

def get_relative_readability(readability_dict):
    updated_variants = cal_relative_readability(readability_dict)
    return updated_variants

def get_relative_complexity(complexity_dict):
    updated_variants = cal_relative_complexity(complexity_dict)    
    return updated_variants

def cal_relative_readability(variants): # to return relative readability of the latter one
    for variant_file in variants:
        diff_readability_weight = {}
        relative_readability = 0
        diff_readability = {}
        variant = variants[variant_file]
        readability = variant["readability"]
        for key in readability:
            if key in ["File", "Log"]:
                continue
            diff = abs(float(readability[key]) - (min([float(variants[v]["readability"][key]) for v in variants])))
        if key in ["itid_rate", "nmi", "average_tc"]:
            diff_readability[key] = formula(diff, 0)
        else:
            diff_readability[key] = formula(diff, 1)
        for key in diff_readability:
            diff_readability_weight[key] = 1/len(diff_readability) # change weight here if needed
            relative_readability += diff_readability_weight[key] * diff_readability[key]
        variants[variant_file]["relative_readability"] = relative_readability
    return variants

def cal_relative_complexity(variants):
    for variant_file in variants:
        diff_complexity_weight = {}
        relative_complexity = 0
        diff_complexity = {}
        variant = variants[variant_file]
        complexity = variant["complexity"]
        for key in complexity:
            if key in ["File", "Log"]:
                continue
            diff = abs(float(complexity[key]) - (min([float(variants[v]["complexity"][key]) for v in variants])))
            diff_complexity[key] = formula(diff, 0)
        for key in diff_complexity:
            diff_complexity_weight[key] = 1/len(diff_complexity) # change weight here if needed
            relative_complexity += diff_complexity_weight[key] * diff_complexity[key]
        variants[variant_file]["relative_complexity"] = relative_complexity
    return variants
    
def formula(diff, r):
    return r * np.exp(-np.log10(1 + diff)) + (1 - r) * (1 - np.exp(-np.log10(1 + diff)))

def cosine_distance_readability(target_dict, result_dict):
    target = target_dict.copy()
    result = result_dict.copy()
    # for key in target:
    #     if key in ["itid_rate", "nmi", "average_tc"]: # the higher, the better
    #         if result[key] > target[key]:
    #             result[key] = target[key]
    #     else: # the lower, the better
    #         if result[key] < target[key]: 
    #             target[key] = result[key]
    
    metric1_vector = np.array(list(target.values()))
    metric2_vector = np.array(list(result.values()))
    print(metric1_vector, metric2_vector)
    cosine_similarity = np.dot(metric1_vector, metric2_vector) / (
        np.linalg.norm(metric1_vector) * np.linalg.norm(metric2_vector))
    return cosine_similarity

def cosine_distance_complexity(target_dict, result_dict):
    target = target_dict.copy()
    result = result_dict.copy()
    for key in target:
        if result[key] > target[key]:
            result[key] = target[key]
    metric1_vector = np.array(list(target.values()))
    metric2_vector = np.array(list(result.values()))

    cosine_similarity = np.dot(metric1_vector, metric2_vector) / (
        np.linalg.norm(metric1_vector) * np.linalg.norm(metric2_vector))
    return cosine_similarity

def relevant_distance_complexity(target_dict, result_dict):
    print(target_dict, result_dict)
    target = target_dict.copy()
    result = result_dict.copy()
    weights = {}
    delta = {}
    features = 0
    for key in target:
        if float(target[key]) == 0:
            continue
        features += 1
        # if result[key] == target[key] or result[key] > target[key]:
        #     delta[key] = 1
        # else:
        delta[key] = result[key]/target[key]
        # weights[key] = 
    # print(delta)
    result = (1/features) * sum([delta[item] for item in delta])
    # print(result)
    return round(result,4)

def relevant_distance_readability(target_dict, result_dict):
    target = target_dict.copy()
    result = result_dict.copy()
    delta = {}
    features = 0
    for key in target:
        if float(target[key]) == 0:
            continue
        features += 1
        # if result[key] == target[key] or result[key] > target[key]:
        #     delta[key] = 0
        # else:
        #     
        delta[key] = 1 - result[key]/target[key]
    # print(delta)
    result = (1/features) * sum([delta[item] for item in delta])
    # print(result)
    return round(result,4)

def get_python_files_dir(directory):
    print(directory)
    file_list = []
    for dir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(dir, file)
                file_list.append(file_path)
    return file_list