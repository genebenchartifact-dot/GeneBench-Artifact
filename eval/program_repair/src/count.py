import os
import json


def count_passed_tests_in_dir(directory):
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        fpath = os.path.join(directory, filename)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Skipping {filename}: {e}")
            continue

        passed = 0
        total = 0
        for instance_id, content in data.items():
            total += 1
            result = content.get("result")

            if isinstance(result, list) and any("pass" in r.lower() or "success" in r.lower() for r in result):
                passed += 1
        print(f"{filename}: {passed}/{total} instances passed")


if __name__ == "__main__":
    directory = "/home/ubuntu/GeneBenchCode/program_repair/Result_icl_fixed_new/"  # <-- replace this with your directory path
    count_passed_tests_in_dir(directory)
