import ast
import csv
import json
import os
import sys
import subprocess

def write_dict_csv(csv_path, fields, dict_data):
    with open(csv_path, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerow(dict_data)

def write_header_csv(csv_path, fields):
    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

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

def write_json(file_path, dict):
    with open(file_path, 'w') as fp:
        json.dump(dict, fp)

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
    diff_opts = ["diff", source_file, target_file]
    diff_output = run_cmds(diff_opts, None)
    write_file(diff_path, diff_output)
    # print(diff_output)
    return diff_output, diff_path

def run_cmds(cmd_list, timeoutVal):
    cmds = " ".join(cmd_list)
    # print(cmds, flush=True)
    if timeoutVal != None:
        run_cmds = subprocess.run(cmd_list, stdout=subprocess.PIPE, timeout=timeoutVal) #check=True, capture_output=True, shell=True
    else:
        run_cmds = subprocess.run(cmd_list, stdout=subprocess.PIPE)
    output = run_cmds.stdout.decode('utf-8')
    # print(output, flush=True)
    return output

def run_cmds_nopipe(cmd_list, timeoutVal):
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

# move here
# git checkout
# print log