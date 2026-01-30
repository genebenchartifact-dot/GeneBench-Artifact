import argparse
import os
import re
import json
import subprocess
from subprocess import Popen, PIPE
from typing import Dict, Any, Optional
import os
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    import utils
    read_file = utils.read_file
    write_file = utils.write_file
    write_json = utils.write_json
except Exception:
    def read_file(path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def write_file(path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def write_json(path: str, data: Any) -> None:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_prompt(code: str) -> str:
    return (
        "I have the following buggy code:\n"
        f"{code}\n"
        "Can you fix it so it passes the tests described in the comments.\n"
    )
    
model_json = {
    "deepseek-ai/deepseek-coder-33b-instruct": "deepseek-coder-33b-instruct",
    "codellama/CodeLlama-13b-Instruct-hf": "CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-13b-hf": "CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-Instruct-hf": "CodeLlama-34b-hf",
    "WizardLM/WizardCoder-15B-V1.0": "WizardCoder-15B-V1.0",
    "bigcode/starcoder2-15b": "starcoder2-15b",
    "semcoder/semcoder": "semcoder_1030",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "deepseek-coder-6.7b-instruct",
    "deepseek-ai/deepseek-coder-6.7b-base": "deepseek-coder-6.7b-base"
    
}

def get_icl_example(ID: str, code: str, model_name: str) -> str:
    if "ClassEval" in ID:
        jsonl_file = "program_repair/icl/classeval_ids.jsonl"
        icl_file = "program_repair/GA_v0_classeval.json"
    elif "HumanEval" in ID:
        jsonl_file = "program_repair/icl/humaneval_ids.jsonl"
        icl_file = "program_repair/GA_v0_humaneval.json"
    icl_id = get_icl_id(ID, jsonl_file, model_name)
    icl_code = get_icl_code(icl_id, icl_file)
    return icl_code

def get_icl_code(icl_id, json_file):
    json_data = read_json(json_file)
    return json_data[icl_id]['transformation']

def get_icl_id(ID: str, jsonl_file: str, model_name: str) -> str:
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['code'] == ID:
                model = model_json[model_name]
                return data['ICL example'][model]
            
from openai import OpenAI
class LLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_openai = "gpt" in model_name or "o4-mini" in model_name
        self.is_deepseek = "deepseek-reasoner" in model_name
        self.is_vllm = not (self.is_openai or self.is_deepseek)

        self.openai_client = None
        self.deepseek_client = None
        self.vllm_model = None
        self.sampling = None

        self._init_clients()

    def _init_clients(self):
        if self.is_openai:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.is_deepseek:
            self.deepseek_client = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
        else:
            device = '0,1'
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                # token=os.getenv("AUTH_TOKEN"),
            )
            self.vllm_model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                max_model_len=8000,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.90,
                dtype="bfloat16"
            )
            self.sampling = SamplingParams(
                max_tokens=4096,
                temperature=0.0,
            )

    def _flatten_messages(self, messages):
        system = "\n".join([m["content"] for m in messages if m["role"] == "system"])
        user = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        return f"{system}\n\nUser:\n{user}".strip()

    def chat(self, prompt: str, timeout_sec: int = 240) -> str:
        messages = [
            {"role": "system", "content": "You are an expert Python programmer and assistant."},
            {"role": "user", "content": prompt},
        ]

        if self.is_openai and self.openai_client:
            resp = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()

        elif self.is_deepseek and self.deepseek_client:
            resp = self.deepseek_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                # temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()

        else:
            assert self.vllm_model is not None and self.sampling is not None
            flat_prompt = self._flatten_messages(messages)
            vllm_outputs = self.vllm_model.generate([flat_prompt], self.sampling)
            if vllm_outputs and vllm_outputs[0].outputs:
                return vllm_outputs[0].outputs[0].text.strip()
            return ""

FENCE_PATTERNS = [
    r"```python\s*(.*?)```",
    r"\[PYTHON\]\s*(.*?)\[/PYTHON\]",
]

def extract_code_from_response(response: str) -> str:
    if not response:
        return ""
    if "```python" in response:
        response = response.split("```python")[-1].strip().split("```")[0]
    elif "[PYTHON]" in response:
        response = response.split("[PYTHON]")[-1].strip().split("[/PYTHON]")[0]
    if "# Test" in response:
        response = response.split("# Test")[0].strip()
    if "print" in response:
        # remove parts starting from print, include print
        response = re.split(r"\bprint\s*\(", response, maxsplit=1)[0].strip()

    # for pat in FENCE_PATTERNS:
    #     m = re.search(pat, response, flags=re.DOTALL | re.IGNORECASE)
    #     if m:
    #         return m.group(1).strip()
    # if "```" in response:
    #     return response.split("```")[-1].strip()
    return response.strip()


def extract_and_write_dependent_files(code: str, tmp_dir: str) -> None:
    """
    Scan the code string for dependent file blocks and write them to tmp_dir.
    """
    # if "#The following is code in" in code:
    #     filename = code.split("#The following is code in dependent file ")[-1].split("\n")[0].strip()
    #     content = code.split(f"{filename}\n")[-1].strip()
    #     file_path = os.path.join(tmp_dir, filename)
    #     write_file(file_path, content)
    #     # exit(0)
    # elif "# code in " in code:
    #     filename = code.split("# code in")[-1].split("\n")[0].strip()
    #     content = code.split(f"{filename}\n")[-1].strip()
    #     file_path = os.path.join(tmp_dir, filename)
    #     write_file(file_path, content)
    new_lines = []
    for line in code.splitlines():
        if "from " in line and "import " in line and " as " in line:
            new_lines.append(f"# {line}") 
        else:
            new_lines.append(line)
    code = "\n".join(new_lines)
    return code



def run_classeval_test(id_: str, code: str, tests_dir: str) -> list:
    print(f"*Running ClassEval tests for: {id_}")
    test_file = os.path.join(tests_dir, "test", id_ + "_test.py")
    input_content = read_file(test_file)

    entry_point = (
        "\nif __name__ == '__main__':\n"
        "    import unittest\n"
        "    unittest.main()\n"
    )

    tmp_dir = ".tmp_test"
    os.makedirs(tmp_dir, exist_ok=True)
    check_file = os.path.join(tmp_dir, f"tmp_{id_}.py")
    write_file(check_file, f"{code}\n{input_content}\n{entry_point}")

    failure, tests_pass, error_or_timeout = [], [], []
    try:
        result = subprocess.run(
            ["python3", check_file],
            capture_output=True, text=True, timeout=60
        )
        out = (result.stdout or "") + (result.stderr or "")
        print(out)
        if "OK" in out and "FAILED" not in out and "Traceback" not in out:
            tests_pass.append(id_)
        elif "AssertionError" in out or "FAIL" in out or "FAILED" in out:
            failure.append(id_)
        else:
            error_or_timeout.append(id_)
    except subprocess.TimeoutExpired:
        error_or_timeout.append(id_)

    return [k for k, v in {
        "tests_pass": tests_pass,
        "failure": failure,
        "error_or_timeout": error_or_timeout
    }.items() if v]

def run_humaneval_test(id_: str, code: str, tests_dir: str) -> list:
    print(f"*Running HumanEval tests for: {id_}")
    test_file = os.path.join(tests_dir, "test", id_ + "_test.py")
    entry_file = os.path.join(tests_dir, "entry", id_ + "_entry.txt")

    input_content = read_file(test_file)
    entry_point = read_file(entry_file)

    tmp_dir = ".tmp_test"
    os.makedirs(tmp_dir, exist_ok=True)
    check_file = os.path.join(tmp_dir, f"tmp_{id_}.py")
    write_file(check_file, f"{code}\n{input_content}\ncheck({entry_point})")

    failure, tests_pass, error_or_timeout = [], [], []
    try:
        p = Popen(["python3", check_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate(timeout=10)
        out, err = stdout.decode("utf-8"), stderr.decode("utf-8")
        print(out, err)

        if out.strip() == "" and err.strip() == "":
            tests_pass.append(id_)
            print(f"{id_} pass!")
        elif "AssertionError" in out or "AssertionError" in err:
            failure.append(id_)
            print(f"{id_} fail!")
        else:
            error_or_timeout.append(id_)
            print(f"{id_} error!")
    except subprocess.TimeoutExpired:
        error_or_timeout.append(id_)

    return [k for k, v in {
        "tests_pass": tests_pass,
        "failure": failure,
        "error_or_timeout": error_or_timeout
    }.items() if v]

def run_pytest_for_id(id_: str, code: str, tests_dir: str) -> list:
    tmp_dir = ".tmp_test"
    os.makedirs(tmp_dir, exist_ok=True)

    # write supporting files (e.g., utility_functions.py, etc.)
    code = extract_and_write_dependent_files(code, tmp_dir)

    # delegate to appropriate test runner
    if "HumanEval" in id_:
        return run_humaneval_test(id_, code, tests_dir)
    if "ClassEval" in id_:
        return run_classeval_test(id_, code, tests_dir)
    raise ValueError(f"{id_} is not a supported benchmark id.")


from tqdm import tqdm

import ast
import random

def extract_random_test_functions(source_code: str, k=1) -> str:
    tree = ast.parse(source_code)
    test_funcs = []

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    func_code = ast.get_source_segment(source_code, item)
                    test_funcs.append(func_code)

    picked = random.sample(test_funcs, k=min(k, len(test_funcs)))
    return "\n\n".join(picked)


def add_tests(code, ID, tests_dir):
    if "ClassEval" in ID:
        prompt_file = os.path.join(tests_dir, "prompt", ID + ".txt")
        prompt_content = read_file(prompt_file)
        hint = prompt_content.split("\"\"\"")[1]
        # tests = []
        # for line in prompt_content.splitlines():
        #     line = line.strip()
        #     if line.startswith(">>>"):
        #         tests.append(line)
        #     if len(tests) > 3:
        #         break
        # tests_str = "\n".join(tests)
        
        test_file = os.path.join(tests_dir, "test", ID + "_test.py")
        test_content = read_file(test_file)
        tests = extract_random_test_functions(test_content)
        new_code = f"\"\"\"\n{hint}\nTests:\n{tests}\n\"\"\"\n{code}"
        return new_code
    return code

def main(tests_dir: str,
         model_name: str,
         output_path: str,
         bug_info_path: str,
         datasetname: str,
        icl=False):

    print(f"[INFO] model={model_name} dataset={datasetname}")

    dataset = read_json(bug_info_path)
    results: Dict[str, Dict[str, Any]] = {}
    client = LLMClient(model_name)

    if client.is_vllm:
        id_list, prompts = [], []

        for ID, buggy_code in dataset.items():
            if buggy_code:
                buggy_code = add_tests(buggy_code, ID, tests_dir)
                base_prompt = build_prompt(buggy_code)
                
                if icl:
                    icl_example = get_icl_example(ID, buggy_code, model_name)
                    base_prompt += f"\nThe following program may help you think:\n{icl_example}\n"
                
                if "deepseek" in model_name or "wizardcoder" in model_name or "semcoder" in model_name or "WizardCoder" in model_name:
                    prompt = f"###Instructions\n{base_prompt}\nEnclose only MAIN code in ```python ... ``` Do not include explanations or any tests. Do not add any additional prints.\n### Response\n"
                elif "codellama" in model_name or "CodeLlama" in model_name:
                    prompt = f"[INST]Prompt\n{base_prompt}\nEnclose your code in [PYTHON] and [/PYTHON]\n[/INST]\n"
                elif "starcoder" in model_name:
                    prompt = f"<fim_prefix>\n{base_prompt}\nEnclose your code in [PYTHON] and [/PYTHON]\nResponse:<fim_suffix><fim_middle>\n"
                else:
                    prompt = base_prompt + "Enclose all code in ```python ... ```  (no explanations, no tests)."
                
                id_list.append(ID)
                prompts.append(prompt)

        batch_size = 32

        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating (vLLM batch)"):
            batch_prompts = prompts[i:i + batch_size]
            batch_ids = id_list[i:i + batch_size]

            vllm_outputs = client.vllm_model.generate(batch_prompts, client.sampling)

            for j, output in enumerate(vllm_outputs):
                ID = batch_ids[j]
                prompt = batch_prompts[j]
                response = "\n".join(o.text for o in output.outputs) if output.outputs else ""
                print(f"ID: {ID}")
                print(f"Prompt:\n{prompt}")
                print(f"Response:\n{response}")
                extracted_code = extract_code_from_response(response)
                test_result = run_pytest_for_id(ID, extracted_code, tests_dir)

                results[ID] = {
                    "prompt": prompt,
                    "response": response,
                    "code": extracted_code,
                    "result": test_result,
                }

    else:
        for ID, buggy_code in tqdm(dataset.items(), desc="Generating (sequential)"):
            if not buggy_code:
                continue

            buggy_code = add_tests(buggy_code, ID, tests_dir)
            prompt = build_prompt(buggy_code)
            # icl_example = get_icl_example(ID, buggy_code)
            # prompt += f"\nThe following program may help you think:\n{icl_example}\n"
            response = client.chat(prompt)
            extracted_code = extract_code_from_response(response)
            test_result = run_pytest_for_id(ID, extracted_code, tests_dir)
            
            print(f"ID: {ID}")
            print(f"Prompt:\n{prompt}")
            print(f"Response:\n{response}")
            print(f"Test Result:\n{test_result}")

            results[ID] = {
                "prompt": prompt,
                "response": response,
                "code": extracted_code,
                "result": test_result,
            }

    # Write results
    write_json(output_path, results)
    try:
        resp_path = output_path.replace("Result", "response")
        write_json(resp_path, results)
    except Exception:
        pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="program repair eval."
    )

    parser.add_argument("--tests-dir", required=True,
                        help="Directory containing test files")
    parser.add_argument("--model-name", required=True,
                        help="Model name")
    parser.add_argument("--output-path", required=True,
                        help="Path to save output JSON with results")
    parser.add_argument("--bug-info-path", required=True,
                        help="Path to JSON containing buggy code info")
    parser.add_argument("--datasetname", choices=["HumanEval", "ClassEval"], required=True,
                        help="Dataset name")

    args = parser.parse_args()

    main(
        tests_dir=args.tests_dir,
        model_name=args.model_name,
        output_path=args.output_path,
        bug_info_path=args.bug_info_path,
        datasetname=args.datasetname,
    )
