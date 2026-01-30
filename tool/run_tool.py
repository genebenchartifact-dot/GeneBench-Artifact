import json
import csv
import ast
import datetime
from rope.refactor.extract import ExtractVariable
from rope.base import libutils
import rope.base.project
import code_refactor
import pylint.lint
import subprocess
import utils
import traceback
import os
import shutil
from subprocess import Popen, PIPE

python_samples = []
infinite_loop = []
compilation_err = []
consistent = []
pass_items = []

def read_file(file_path):
    file = open(file_path, 'r')
    content = file.read()
    return content

def write_file(file_path,content):
    f = open(file_path, "w")
    f.write(content)
    f.close()

def read_json(json_path):
    f = open(json_path)
    data = json.load(f)
    all_info = []
    for items in data:
        if items["language"] == "Python":
            python_samples.append(items)
    return python_samples

def run_item(item, file_path, before_after):
    if item["language"] == "Python":
        if len(item["test_IO"]) == 1:
            input = item["test_IO"][0]["input"]
            original_output = item["test_IO"][0]["output"]
            print("original output:",original_output.strip(), flush=True)
            py_output = run_py_code(file_path,input,item["id"]).replace("\n"," ").strip()
            print("output_" + before_after + "_refactoring:",py_output.strip(), flush=True)
        else:
            py_output = []
            for instance in item["test_IO"]:
                input = instance["input"]
                py_output_once = run_py_code(file_path,input,item["id"]).replace("\n"," ").strip()
                py_output.append(py_output_once)
                print("output_" + before_after + "_refactoring:",py_output_once.strip(), flush=True)
    return " ".join(py_output)

def run_py_code(file_path,input,id):
    try:
        py_opts =  ["python3", file_path]
        print(" ".join(py_opts), flush=True)
        p = Popen(py_opts, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            stdout, stderr_data = p.communicate(input=input.encode(), timeout=100)
        except subprocess.TimeoutExpired:
            infinite_loop.append(id)
            print(id, "infinite_loop", flush=True)
        py_output = stdout.decode('utf-8')
    except Exception as e:
        compilation_err.append(id)
        print(id, "compile_failed", e, flush=True)
    return py_output

def run_sourcery(file_path):
    try:
        refactor_types = []
        git_checkout_output = utils.git_checkout(file_path)

        log_path = file_path.replace(".py","_log.log")
        sourcery_opts = ["python3", "tool/init.py", "-s", file_path, "-t", file_path, "-e", "/home/yang/contamination/dataset/avatar/Python/TestCases"]
        sourcery_output = run_cmds(sourcery_opts, None)

        codebleu = None
        final = None
        for line in sourcery_output.split("\n"):
            if "is applicable" in line and "Rule" in line:
                refactor_types.append(line.split(" ")[1])
            if "codebleu" in line and "ngram_match_score" in line:
                codebleu = line.split(",")[0].split(":")[1].strip()
            if "Final tests:" in line:
                final = line.split(":")[-1].strip()
        print(sourcery_output, log_path, codebleu)
        write_file(log_path, sourcery_output)

        diff_path = git_diff(file_path)
        print(diff_path)
        print(refactor_types)
        if refactor_types != []:
            shutil.copy(file_path, "/home/yang/contamination/refactored_avatar_python")
        return diff_path,log_path,refactor_types, codebleu,final
    except Exception as e:
        print(file_path,e)

def cal_cyclomatic_cmplx(file_path):

    radon_halstead_opts =  ["radon", "hal", file_path]
    hal = subprocess.run(radon_halstead_opts, stdout=subprocess.PIPE)
    hal_output = hal.stdout.decode('utf-8')

    radon_mi_opts = ["radon", "mi", file_path, "-s"]
    mi = subprocess.run(radon_mi_opts, stdout=subprocess.PIPE)
    mi_output = mi.stdout.decode('utf-8')
    mi_value = mi_output.split(" - ")[-1].replace(" ","").replace("\n","")
    return mi_value

def main(items, save_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_file, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile)
        fields = ["id", "file_name","diff_path", "log_path",
            "mi_value_before", "mi_value_after",
             "passFlag","refactor_types", "codebleu"]
        csvwriter.writerow(fields)
    for item in items:
        with open(save_file, 'a') as csvfile:  
            csvwriter = csv.writer(csvfile)
            passFlag = False
            file_name = save_dir + item["id"] + ".py"
            write_file(file_name, item["code"])

            print("start_process_of",item["id"], flush=True)
            print("before_refactor", item["id"], flush=True)
            mi_value_before = cal_cyclomatic_cmplx(file_name)
            print("mi_value_before", mi_value_before, flush=True)

            print("refactoring", item["id"], flush=True)
            diff_path, log_path, refactor_types, codebleu, final = run_sourcery(file_name)
            refactored_code = read_file(file_name)
            write_file(file_name.replace("_before.py","_after.py"), refactored_code)

            print("after_refactor", item["id"], flush=True)
            mi_value_after = cal_cyclomatic_cmplx(file_name)
            print("mi_value_after", mi_value_after, flush=True)
            print("end_process_of",item["id"],"\n", flush=True)
            csvwriter.writerow([item["id"],file_name, 
                diff_path, log_path, mi_value_before, 
                mi_value_after, final, refactor_types, codebleu])
            # exit(0)
            


if __name__ == "__main__":
    # args = sys.argv[1:]
    # input_file = args[0] 
    input_file ="/home/yang/contamination/avatar/avatar.json"
    save_file = "avatar_python.csv"
    save_dir = "Avatar_Python/"
    #  "/home/yang/contamination/QuixBugs/python_programs/breadth_first_search.py"
    python_samples = read_json(input_file)
    print("STARTING AT", datetime.datetime.now(), flush=True)
    main(python_samples,save_file)
    print("END AT", datetime.datetime.now(), flush=True)