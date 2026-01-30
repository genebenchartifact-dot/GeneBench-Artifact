import os
import sys
import ast
import json
import argparse
import logging
from litellm import completion
import google.generativeai as genai

from complexity import run_pytest
import utils

os.environ['GEMINI_API_KEY'] = "AIzaSyBek3uiwaZhGXJkSRWC9-unsEos7rnmGAg"


def init_logger(new_dir, instance_id):
    log_dir = f"{new_dir}/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{instance_id}.log")

    logger = logging.getLogger(instance_id)
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    logger.addHandler(handler)
    return logger


class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Load, ast.Store)) and node.id not in {"True", "False", "None"}:
            if ("new" in node.id and "_" in node.id) or "loopchecker" in node.id or "conditionchecker" in node.id or "Checker" in node.id \
                or "checker" in node.id or "loop_" in node.id or "variable_" in node.id:
                self.variables.add(node.id)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        if node.name not in {"True", "False", "None"}:
            if "Func_" in node.name:
                self.variables.add(node.name)
        self.generic_visit(node)

    
    def visit_Module(self, node):
        self.generic_visit(node)


def extract_variables(code, logger):
    tree = ast.parse(code)
    collector = VariableCollector()
    collector.visit(tree)
    logger.info(f"Extracted variables: {collector.variables}")
    return collector.variables


def replace_variable_in_statement(code, target_var, replacement_var):
    try:
        return code.replace(target_var, replacement_var)
    except Exception as e:
        return code


def read_file(path):
    with open(path, 'r') as f:
        return f.read()


def generator_gemini(prompt, logger, model="gemini-1.5-pro"):
    response = completion(
        model=f"gemini/{model}",
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_object",
            "response_schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "original name": {"type": "string"},
                        "variable name": {"type": "string"},
                    },
                    "required": ["variable name"],
                }
            }
        }
    )
    content = response.choices[0].message.content
    logger.info(f"Gemini response: {content}")
    return json.loads(content)


def get_newclass_imports(code):
    class_funcs = {}
    if "from newClass" in code:
        for line in code.split("\n"):
            if "from newClass" in line and "import" in line:
                newclass = line.split("from")[-1].split("import")[0].strip()
                alias = line.split("as")[-1].strip()
                class_funcs.setdefault(newclass, []).append(alias)
    return class_funcs


def process_single_file(file_path, file, dirpath, dataset, tests_dir, new_dir, items, ID):
    logger = init_logger(new_dir, ID)
    code = read_file(file_path)
    renamed_code_dir = f"{new_dir}/renamed_code"
    check_file = os.path.join(renamed_code_dir, f"{ID}.py")

    if os.path.exists(check_file):
        logger.info(f"{check_file} exists")
        return

    logger.info(f"Processing {ID}")
    logger.info(f"File path: {file_path}")
    logger.info(f"Original code:\n{code}")
    class_funcs = get_newclass_imports(code)
    logger.info(f"Class functions:\n{class_funcs}")
    vars_to_rename = extract_variables(code, logger)
    logger.info(f"Variables to rename:\n{vars_to_rename}")
    prompt = f"Can you generate new natural and readable names for the following variables? Make sure after renaming, the original code is correct and functionality is preserved. Do not include any numbers in the new name. Make sure the new names are human readable. Make sure class names like newClass are changed correctly.\n{vars_to_rename} {list(class_funcs.keys())}\n{code}"
    logger.info(f"Prompt for Gemini:\n{prompt}")
    answers = generator_gemini(prompt, logger)
    logger.info(f"Gemini response:\n{answers}")

    vas_list = list(vars_to_rename)
    idx = 0
    for pair in answers:
        if "original name" not in pair:
            pair["original name"] = vas_list[idx]
        idx += 1

    #process code
    for pair in answers:
        if pair["original name"] in class_funcs or pair["original name"] in vars_to_rename:
            code = replace_variable_in_statement(code, pair["original name"], pair["variable name"])
    logger.info(f"Processed code:\n{code}")
    if os.path.exists(".tmp.py"):
        os.remove(".tmp.py")
    utils.write_file(".tmp.py", code)

    items[file] = {
        "ID": ID,
        "file": file,
        "original_code": read_file(file_path),
        "file_path": file_path,
        "names": answers,
        "class_funcs": class_funcs,
        "new_code": None,
        "newclass_code": None,
        "prompt": None,
        "original_prompt": None
    }

    if class_funcs:
        process_with_newclass(code, class_funcs, dirpath, answers, ID, file, dataset, tests_dir, new_dir, items, logger)
    else:
        process_without_newclass(code, ID, file, tests_dir, new_dir, items, logger)


def process_with_newclass(code, class_funcs, dirpath, answers, ID, file, dataset, tests_dir, new_dir, items, logger):
    logger.info("process_with_newclass")
    for classname in class_funcs:
        class_code_path = os.path.join(dirpath, f"{classname}.py")
        if not os.path.exists(class_code_path):
            continue
        class_code = read_file(class_code_path)
        new_class_code = class_code

        for pair in answers:
            if pair["original name"] in class_funcs[classname]:
                new_class_code = replace_variable_in_statement(new_class_code, pair["original name"], pair["variable name"])

        if "newClass" in code:
            logger.info(f"New class {classname} found in code")
            continue

        if classname == pair["original name"]:
            classname = pair["variable name"]

        utils.write_file(f".tmp_test/{classname}.py", new_class_code)
        logger.info(f"Run tests")
        res = run_pytest(ID, ".tmp.py", tests_dir)
        logger.info(f"Pytest result: {res}")

        if "tests_pass" in res:
            save_successful_rename(ID, file, code, new_class_code, classname, new_dir, items, class_code, logger)
        else:
            logger.info(f"{ID} fails after renaming!")


def process_without_newclass(code, ID, file, tests_dir, new_dir, items, logger):
    res = run_pytest(ID, ".tmp.py", tests_dir)
    logger.info(f"Pytest result: {res}")
    if "tests_pass" in res:
        logger.info(f"{ID} pass after renaming!")
        renamed_code_dir = f"{new_dir}/renamed_code"
        os.makedirs(renamed_code_dir, exist_ok=True)
        utils.write_file(os.path.join(renamed_code_dir, f"{ID}.py"), code)
        utils.write_file(os.path.join(new_dir, f"prompts/{ID}.py"), code)
        items[file]["new_code"] = code
        items[file]["prompt"] = code
        items[file]["original_prompt"] = items[file]["original_code"]
    else:
        logger.info(f"{ID} fails after renaming")


def save_successful_rename(ID, file, code, new_class_code, classname, new_dir, items, class_code, logger):
    logger.info(f"{ID} pass after renaming!")
    renamed_code_dir = f"{new_dir}/renamed_code"
    os.makedirs(renamed_code_dir, exist_ok=True)
    utils.write_file(os.path.join(renamed_code_dir, f"{ID}.py"), code)
    utils.write_file(os.path.join(renamed_code_dir, f"{classname}.py"), new_class_code)
    prompt_text = f"{code}\n\n#The following is code in dependent file {classname}.py:\n{new_class_code}"
    utils.write_file(os.path.join(new_dir, f"prompts/{ID}.py"), prompt_text)

    items[file]["new_code"] = code
    items[file]["newclass_code"] = new_class_code
    items[file]["prompt"] = prompt_text
    items[file]["original_prompt"] = f"{items[file]['original_code']}\n#The following is code in dependent file {classname}.py:\n{class_code}"


def get_files(file_dir, dataset, tests_dir, new_dir):
    print(f"Working on {dataset}.")
    items = {}

    for dirpath, _, files in os.walk(file_dir):
        for file in files:
            if ".py-" in file and "newClass" not in file:
                file_path = os.path.join(dirpath, file)
                ID = file.split(".py-")[0]
                process_single_file(file_path, file, dirpath, dataset, tests_dir, new_dir, items, ID)

    utils.write_json(f"{new_dir}/renamed_jsons/{dataset}_renaming.json", items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variable Renaming with Gemini and Pytest Verification")
    parser.add_argument("--file_dir", required=True, help="Directory containing original Python files")
    parser.add_argument("--dataset", required=True, help="Dataset name (used for JSON output naming)")
    parser.add_argument("--tests_dir", required=True, help="Directory containing test files")
    parser.add_argument("--new_dir", required=True, help="Directory to save renamed files and prompts")
    args = parser.parse_args()

    os.makedirs(args.new_dir, exist_ok=True)
    os.makedirs(os.path.join(args.new_dir, "prompts"), exist_ok=True)

    get_files(args.file_dir, args.dataset, args.tests_dir, args.new_dir)
