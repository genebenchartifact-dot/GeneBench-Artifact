import csv
import datetime
import json
import os
import sys
import utils
import complexity

"""
To generate json files in the following format:
{ #"src_language":None,"target_language":None,"result_before":None,"result_after":None,

    "id":None, "dataset":None, "language": None
    "filepath_before":None, "filepath_after":None,
    "code_before":None, "code_after":None,
    "LOC_before":None, "LOC_after":None,
    "MI_before":None, "MI_after":None,
    "CC_before":None, "CC_after":None,
    "HM_difficulty_before":None, "HM_difficulty_after":None,
    "HM_effort_before":None, "HM_effort_after":None,
    "transformation_types":None, 
    "patchpath": None, "patch_LOC": None,
}
"""

def match_instances(dir):
    result_dict = {}
    idx = 0
    for dirpath, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".py"):
                idx += 1
                filepath = os.path.join(dirpath, file)
                print(dir, idx, filepath)
                id = file.split(".")[0]
                code = utils.read_file(filepath)
                res = complexity.cal_complexity(filepath)
                cc = complexity.cal_cyclomatic_cmplx(filepath)
                if id not in result_dict:
                    result_dict[id] ={
                        "id": id, "filepath": filepath, "code": code, 
                    }
                result_dict[id] = result_dict[id] | res
                result_dict[id] = result_dict[id] | cc
    return result_dict

def combine_before_After(dir_before, dir_after, lang, dataset):
    final_dict = {}
    idx = 0
    for id in dir_after:
        idx += 1
        print("combine", idx)
        after_info = dir_after[id]
        new_instance = {}
        if id not in dir_before:
            raise Exception("Not Match" + id)
        
        before_info = dir_before[id]

        if id not in final_dict:
            final_dict[id] = {"id":id, "language":lang, "dataset": dataset, "pass_before": 0, "pass_after": 0}
        
        for key in before_info:
            if key == "id":
                continue
            if key + "_before" not in final_dict[id]:
                final_dict[id][key + "_before"] = before_info[key]
            if key + "_after" not in final_dict[id]:
                final_dict[id][key + "_after"] = after_info[key]
    return final_dict

def collect_pass_info(dir):
    pass_all = {}
    for dirpath, _, files in os.walk(dir):
        for file in files:
            filepath = os.path.join(dirpath, file)
            if file.endswith(".txt"):
                pass_list = utils.read_file(filepath).split("\n")
                pass_all[file] = pass_list
    before_after = {
        "1": {},
        "2": {},
        "3": {}
        # "1": {"before":None,"after":None, }
    }
    for key in pass_all:
        idx = key.split("_")[-1].split(".")[0]
        if idx not in before_after:
            before_after[idx] = {"before_avatar":None,"before_codenet":None, "after_avatar": None, "after_codenet": None}
        if "latest" not in key:
            if "avatar" in key:
                before_after[idx]["before_avatar"] = pass_all[key]
            else:
                before_after[idx]["before_codenet"] = pass_all[key]
        else:
            if "avatar" in key:
                before_after[idx]["after_avatar"] = pass_all[key]
            else:
                 before_after[idx]["after_codenet"] = pass_all[key]

    for key in before_after:
        for item in before_after[key]["after_avatar"][:]:
            if item not in before_after[key]["before_avatar"]:
                before_after[key]["after_avatar"].remove(item)

        for item in before_after[key]["after_codenet"][:]:
            if item not in before_after[key]["before_codenet"]:
                before_after[key]["after_codenet"].remove(item)
    print("Num, Avatar_Before_Transformation, Avatar_After_Transformation")
    for key in before_after:
        print("{},{},{}".format(key, len(before_after[key]["before_avatar"]),len(before_after[key]["after_avatar"]) ))
    print("Num, Codenet_Before_Transformation, Codenet_After_Transformation")
    for key in before_after:
        print("{},{},{}".format(key, len(before_after[key]["before_codenet"]),len(before_after[key]["after_codenet"]) ))
    return pass_all

def match_info(final_dict, pass_list):
    final = {
        "effort_before":[],
        "effort_after":[],
        "difficulty_before":[],
        "difficulty_after" :[],
        "calculated_length_before":[],
        "calculated_length_after":[],
        "length_before": [],
        "length_after": [],
        "vocabulary_before": [],
        "vocabulary_after": [],
        "mi_value_before": [],
        "mi_value_after": [],
        "Nloc_before": [],
        "Nloc_after": [],
        "AvgCCN_before": [],
        "AvgCCN_after": [],
        # "Fun_Cnt_before": [],
        # "Fun_Cnt_after": []
    }

    for id in final_dict:
        info = final_dict[id]
        for key in pass_list:
            if id in pass_list[key]:
                if "new" in key:
                    info["pass_after"] += 1
                else:
                    info["pass_before"] += 1
        for k in final:
            final[k].append(float(info[k]))
        if info["AvgCCN_before"] != info["AvgCCN_after"]:
            print(id, info["AvgCCN_before"], info["AvgCCN_after"]) 
    
    for k in final:
        print(k, round(sum(final[k])/len(final[k]),2))
    return final_dict
        
if __name__ == "__main__":
    args = sys.argv[1:]
    pass_dir = args[0]

    """
    dir_before = args[0]
    dir_after = args[1]
    match_instances(dir_before, dir_after)

    dir_before = "/home/yang/contamination/Avatar_Python_Before/"
    dir_after = "/home/yang/contamination/Avatar_Python_After/"
    pass_dir = "/home/yang/contamination/tool/pass" #/tool/pass
    save = "avatar_info.json"
    """

    # print("STARTING AT", datetime.datetime.now(), flush=True)

    pass_list = collect_pass_info(pass_dir)
    # print("END AT", datetime.datetime.now(), flush=True)

    """info_before = match_instances(dir_before)
    info_after = match_instances(dir_after)
    lang = dir_before.split("_")[-2]
    dataset = dir_before.split("_")[-3].split("/")[-1]
    final_dict = combine_before_After(info_before, info_after,lang, dataset)
    update_dict = match_info(final_dict, pass_list)
    utils.write_json(save, update_dict)"""
    