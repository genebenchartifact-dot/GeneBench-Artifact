import json
import os

# Set paths
jsonl_path = "/home/yang/cruxeval/data/cruxeval_200.jsonl"
code_dir = "renamed/GA_v1/cruxeval/prompts/"
output_path = "/home/yang/cruxeval/data/cruxeval_200_gav1.jsonl"

# Process the jsonl file
with open(jsonl_path, 'r') as fin, open(output_path, 'w') as fout:
    for line in fin:
        data = json.loads(line)
        code_id = data.get("id")
        if code_id:
            code_file = os.path.join(code_dir, f"{code_id}.py")
            if os.path.exists(code_file):
                with open(code_file, 'r') as cf:
                    new_code = cf.read()
                data["code"] = new_code
        fout.write(json.dumps(data) + '\n')
