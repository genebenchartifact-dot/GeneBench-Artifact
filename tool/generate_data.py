import csv
import os
import sys
import utils

def get_transformed_list(csv_path):
    transformed = []
    codebleu = []
    with open(csv_path, mode ='r') as file:    
        csvFile = csv.reader(file)
        for line in csvFile:
            if "id" in line:
                continue
            if line[2] !="[]":
                transformed.append(line[0])
                codebleu.append(float(line[-1]))
    print(sum(codebleu)/len(codebleu))
    print(csv_path, "#transformed instances", len(transformed))
    return transformed # sum(codebleu)/len(codebleu)

def get_results_before_transformation(log_path, avatar_transformed):
    with open(log_path, 'r') as file:
        data = file.read()
    test_passed = []
    for line in data.split("\n"):
        if line.startswith("Failed Test Files"):
            fail_list = line.split("[")[-1].split("]")[0].strip().replace("'","").replace(" ","").replace(".java", "").split(",")
        elif line.startswith("Runtime Error Files"):
            run_err = line.split("[")[-1].split("]")[0].strip().replace("'","").replace(" ","").replace(".java", "").split(",")
        elif line.startswith("Compilation Error Files"):
            comp_err = line.split("[")[-1].split("]")[0].strip().replace("'","").replace(" ","").replace(".java", "").split(",")
        elif line.startswith("Infinite Loop Files:"):
            infinite = line.split("[")[-1].split("]")[0].strip().replace("'","").replace(" ","").replace(".java", "").split(",")
        elif line.startswith("Test_passed:"):
            test_passed = line.split("[")[-1].split("]")[0].strip().replace("'","").replace(" ","").replace(".java", "").split(",")
    if test_passed != []:
        final_pass = []
        for id in avatar_transformed:
            if id in test_passed:
                final_pass.append(id)
    else:
        final_pass = []
        failed = fail_list + run_err + comp_err + infinite
        for id in avatar_transformed:
            if id not in failed:
                final_pass.append(id)
    return final_pass

if __name__ == "__main__":
    args = sys.argv[1:]
    transformation_avatar = args[0]
    transformation_codenet = args[1]
    numbers_csv = args[2]
    pass_dir = args[3]
    os.makedirs(pass_dir, exist_ok=True)

    # transformation_avatar = "/home/yang/contamination/tool/latest_avatar.csv"
    # transformation_codenet = "/home/yang/contamination/tool/latest_codenet_2.csv"
    avatar_transformed = get_transformed_list(transformation_avatar)
    codenet_transformed = get_transformed_list(transformation_codenet)

    prefix = "/home/yang/PLTranslationEmpirical"
    
    before_dir = [
        "avatar_res_1", "avatar_res_2", "avatar_res_3",
        "codenet_res_1", "codenet_res_2", "codenet_res_3", 
    ]
    results = {
        "before-avatar":[],
        "before-codenet":[],
        "after-avatar":[],
        "after-codenet":[],
    }

    for dir in before_dir:
        if "avatar" in dir:
            file = os.path.join(prefix, dir,"StarCoder_avatar_compileReport_from_Python_to_Java.txt")
            result = get_results_before_transformation(file, avatar_transformed)
            results["before-avatar"].append(len(result))
            print("before transformation avatar", dir, len(result))
            utils.write_file(pass_dir + "/" + dir + ".txt", str(result).replace(" ","").replace("'","").replace("[","").replace("]","").replace(",","\n"))

        else:
            file = os.path.join(prefix, dir,"StarCoder_codenet_compileReport_from_Python_to_Java.txt")
            result = get_results_before_transformation(file, codenet_transformed)
            results["before-codenet"].append(len(result))
            print("before transformation codenet", dir, len(result))
            utils.write_file(pass_dir + "/" + dir + ".txt", str(result).replace(" ","").replace("'","").replace("[","").replace("]","").replace(",","\n"))


    after_dirs = ["latest_avatar_res_1", "latest_avatar_res_2", "latest_avatar_res_3",
    "latest_codenet_res_1", "latest_codenet_res_2", "latest_codenet_res_3"]

    for dir in after_dirs:
        if "avatar" in dir:
            file = os.path.join(prefix, dir,"StarCoder_avatar_compileReport_from_Python_to_Java.txt")
            result = get_results_before_transformation(file, avatar_transformed)
            results["after-avatar"].append(len(result))
            print("after transformation avatar", dir, len(result))
            utils.write_file(pass_dir + "/" + dir + ".txt", str(result).replace(" ","").replace("'","").replace("[","").replace("]","").replace(",","\n"))
        elif "codenet" in dir:
            file = os.path.join(prefix, dir,"StarCoder_codenet_compileReport_from_Python_to_Java.txt")
            result = get_results_before_transformation(file, codenet_transformed)
            results["after-codenet"].append(len(result))
            print("after transformation codenet", dir, len(result))
            utils.write_file(pass_dir + "/" + dir + ".txt", str(result).replace(" ","").replace("'","").replace("[","").replace("]","").replace(",","\n"))
    
    content = "\n"
    # for key in results:
    #     print(key, results[key])
    #     content += str(results[key]).replace("[","").replace("]","") + "\n"
    # print(content)

    for i in range(3):
        res = "{},{},{},{}".format("avatar", i, results["before-avatar"][i], results["after-avatar"][i])
        content += res +"\n"
    for i in range(3):
        res = "{},{},{},{}".format( "codenet", i, results["before-codenet"][i], results["after-codenet"][i])
        content += res +"\n"
    
    utils.write_file(numbers_csv, content)
    