import os
import json
from pathlib import Path
import random
import re
import string

from cv2 import line

def inject_bug(original_line):
    bugs = []
    
    def change_func_args(line):
        match = re.match(r'^(\s*def\s+\w+\s*)\((.*?)\)(\s*(->\s*[^:]+)?\s*:)', line)
        if match:
            prefix, args_str, suffix, _ = match.groups()

            args = [a.strip() for a in args_str.split(',') if a.strip()]
            preserved = []
            if args and args[0] in ('self', 'cls'):
                preserved.append(args[0])

            num_new_args = random.randint(0, 4)
            new_args = [random_arg() for _ in range(num_new_args)]
            full_args = preserved + new_args

            return f"{prefix}({', '.join(full_args)}){suffix}", "ChangeFunctionArgs"
        return None

    def random_arg():
        name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(1, 6)))
        if random.random() < 0.5:
            type_hint = random.choice(['int', 'str', 'float', 'bool', 'List[int]', 'Any'])
            return f"{name}: {type_hint}"
        else:
            return name

    
    def negate_assignment(line):
        match = re.match(r'(\s*[\w\.]+\s*=\s*)(.+)', line)
        if match:
            lhs, rhs = match.group(1), match.group(2)
            if not rhs.strip().startswith('not '):
                return f"{lhs}not {rhs.strip()}", 'NegateAssignment'
        return None
    
    def change_loop_range(line):
        match = re.match(r'(\s*for\s+\w+\s+in\s+)(.+?)(\s*):', line)
        if match:
            prefix, iterable, colon = match.groups()
            if '[:' in iterable or 'range(' in iterable:
                return None  

            n = random.randint(0, 3)  
            return f"{prefix}{iterable}[:{n}]{colon}", 'ChangeLoopRange'
        return None

    def change_arith_operator(line):
        arith_ops = ['+', '-', '*', '/', '%', '**']
        replacements = {'+': '-', '-': '+', '*': '/', '/': '*', '%': '*', '**': '*'}
        for op in arith_ops:
            if f' {op} ' in line:
                return line.replace(f' {op} ', f' {replacements[op]} ', 1), 'ChangeArithOperator'
        return None

    def change_compare_operator(line):
        compare_ops = ['==', '!=', '<=', '>=', '<', '>']
        replacements = {'==': '!=', '!=': '==', '<=': '>', '>=': '<', '<': '>=', '>': '<='}
        for op in compare_ops:
            if f' {op} ' in line:
                return line.replace(f' {op} ', f' {replacements[op]} ', 1), 'ChangeCompareOperator'
        return None

    def change_logical_operator(line):
        logical_ops = [' and ', ' or ']
        replacements = {' and ': ' or ', ' or ': ' and '}
        for op in logical_ops:
            if op in line:
                return line.replace(op, replacements[op], 1), 'ChangeLogicalOperator'
        return None

    def change_return_stat(line):
        match = re.match(r'\s*return\s+(.*)', line)
        if match:
            expr = match.group(1).strip()
            return re.sub(r'return\s+.*', f'return not ({expr})', line), 'ChangeReturnStat'
        return None

    def change_var_type(line):
        match = re.match(r'(\s*\w+\s*=\s*)(\d+)(\s*)$', line)
        if match:
            return match.group(1) + f"'{match.group(2)}'" + match.group(3), 'ChangeVarType'
        return None

    mutation_funcs = [
        change_arith_operator,
        change_compare_operator,
        change_logical_operator,
        change_return_stat,
        change_var_type,
        negate_assignment,
        change_loop_range,
        change_func_args
    ]
    random.shuffle(mutation_funcs)

    mutated_line = original_line
    applied_types = []

    for mutator in mutation_funcs:
        if len(bugs) >= 5:
            break
        result = mutator(mutated_line)
        if result:
            mutated_line, bug_type = result
            bugs.append(bug_type)
            applied_types.append(bug_type)
    if len(applied_types) > 0:
        return {
            "original line": original_line,
            "buggy line": mutated_line,
            "injected bug types": applied_types
        }
    else:
        return None


def get_jsonl_files(dir_path):
    return list(Path(dir_path).rglob("*.jsonl"))

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]

def mutate_bug(original_line):
    res = inject_bug(original_line)
    return res

def collect_info_instance_id(original_classeval, classeval_ga_data):
    common_result = {}
    for inst in classeval_ga_data:
        instance = inst['instance']
        if instance in original_classeval:
            original_code = original_classeval[instance]
            transformation = inst['transformation']
            original_stats = original_code.split("\n")
            transformation_stats = transformation.split("\n")
            common_lines = list(set(original_stats) & set(transformation_stats))
            common_lines = [line for line in common_lines if line != ""]
            buggy_lines = []
            for line in common_lines:
                res = mutate_bug(line)
                if res:
                    buggy_lines.append(res)
            common_result[instance] = {
                "original": original_code,
                "transformation": transformation,
                "common": list(common_lines),
                "buggy": buggy_lines
            }
            if len(common_lines) == 0:
                print(f"No common lines found for instance: {instance}")
                # exit(0)
            if len(buggy_lines) == 0:
                print(f"No buggy lines found for instance: {instance}")
                # exit(0)
    return common_result

def collect_original_code(jsonl_file):
    data = read_jsonl(jsonl_file)
    instance_originalcode = {}
    for inst in data:
        instance = inst['instance']
        original = inst['original code']
        instance_originalcode[instance] = original
    return instance_originalcode

if __name__ == "__main__":
    jsonl_dir = "transformations"
    jsonl_files = get_jsonl_files(jsonl_dir)
    classeval_gas = [
        "transformations/GA_v0/classeval.jsonl",
        "transformations/GA_v1/ClassEval_renamed.jsonl",
        "transformations/GA_v2/ClassEval_renamed.jsonl",
    ]
    humaneval_gas = [
        "transformations/GA_v0/humaneval.jsonl",
        "transformations/GA_v1/HumanEval_renamed.jsonl",
        "transformations/GA_v2/HumanEval_renamed.jsonl"
    ]
    original_humaneval_file = "transformations/GA_v0/humaneval.jsonl"
    original_classeval_file = "transformations/GA_v0/classeval.jsonl"

    original_humaneval = collect_original_code(original_humaneval_file)
    original_classeval = collect_original_code(original_classeval_file)
    
    for classeval_ga in classeval_gas:
        classeval_ga_data = read_jsonl(classeval_ga)
        common_result = collect_info_instance_id(original_classeval, classeval_ga_data)
        # save common_result to json
        new_json_file = "bugs/" + classeval_ga.split("/")[1] + "_classeval_bugs.json"
        with open(new_json_file, "w") as f:
            json.dump(common_result, f, indent=4)

    for humaneval_ga in humaneval_gas:
        humaneval_ga_data = read_jsonl(humaneval_ga)
        common_result = collect_info_instance_id(original_humaneval, humaneval_ga_data)
        new_json_file = "bugs/" + humaneval_ga.split("/")[1] + "_humaneval_bugs.json"
        with open(new_json_file, "w") as f:
            json.dump(common_result, f, indent=4)
