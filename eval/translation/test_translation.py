import os
import subprocess
import pandas as pd
from pathlib import Path
from subprocess import Popen, PIPE
import argparse


def main(args):
    print('testing translations')
    if args.der:
        translation_dir = f"../Experiment_Results/intermediate/DER/Translation/{args.test_type}/{args.model.split('/')[-1]}/{args.dataset}/{args.source_lang}/{args.target_lang}-sanitized"
    else:
        translation_dir = f"../Experiment_Results/intermediate/SR/Translation/{args.test_type}/{args.model.split('/')[-1]}/{args.dataset}/{args.source_lang}/{args.target_lang}-sanitized"
    test_dir = f"../dataset/Intermediate/Translation/{args.dataset}/{args.source_lang}/tests"
    # os.makedirs(args.report_dir, exist_ok=True)
    files = [f for f in os.listdir(translation_dir) if f != '.DS_Store']

    compile_failed = []
    test_passed =[]
    test_failed =[]
    test_failed_details = []
    runtime_failed = []
    runtime_failed_details= []
    infinite_loop = []
    
    if args.target_lang =="Python":

        for i in range(len(files)):

            if not files[i].endswith('.py'):
                continue

            try:
                print('Filename: ', files[i])
                subprocess.run("python3 -m py_compile "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=10)

                with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
                    f_in = f.read()
                
                if args.test_type == "misleading_test":
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_misleading_out.txt", "r").read()
                else:    
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()

                p = Popen(['python3', translation_dir+"/"+ files[i]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                try:
                    stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=10)
                    # print(stdout, stderr_data)
                except subprocess.TimeoutExpired:
                    infinite_loop.append(files[i])
                    continue

                if(stdout.decode().strip()==f_out.strip()):
                    test_passed.append(files[i])
                else:
                    if stderr_data.decode()=='':
                        test_failed.append(files[i])
                        test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
                    else:
                        runtime_failed.append(files[i])
                        runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 

            except Exception as e:
                compile_failed.append(files[i])

    elif args.target_lang =="Java":  # target language 

        for i in range(len(files)):

            if not files[i].endswith('.java'):
                continue
            try:
                print('Filename: ', files[i])
                subprocess.run("javac "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)

                tests_passed = 0
                for j in range(1000):
                    # print("Iter", tests_passed,j)
                    if os.path.exists(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in") == False:
                        if tests_passed == j:
                            test_passed.append(files[i])
                            # print("Pass", files[i])
                        break

                    with open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.in" , 'r') as f:
                        f_in = f.read()
                    f_out = open(test_dir+"/"+ files[i].split(".")[0]+f"_{j}.out", "r").read()
                    p = Popen(['java', files[i].split(".")[0]], cwd=translation_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)

                    try:
                        stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=10)
                        # print(stdout, stderr_data)
                    except subprocess.TimeoutExpired:
                        infinite_loop.append(files[i])
                        break
                    # print(stdout, "stdout", stderr_data)

                    try:
                        if float(stdout.decode())%1 == 0:
                            stdout = str(int(float(stdout.decode())))
                            f_out = str(int(float(f_out)))
                        else:
                            # find how many decimal points are there in the output
                            stdout_temp = stdout.decode().strip()
                            f_out_temp = f_out.strip()
                            f_out_total_dec_points = len(f_out_temp.split(".")[1])
                            stdout_total_dec_points = len(stdout_temp.split(".")[1])
                            min_dec_points = min(f_out_total_dec_points, stdout_total_dec_points)

                            stdout = str(round(float(stdout.decode()), min_dec_points))
                            f_out = str(round(float(f_out), min_dec_points))

                    except:
                        try:
                            stdout = stdout.decode()
                        except:
                            pass
                    # print(stdout.strip(),"v.s.",f_out.strip())
                    if(stdout.strip()==f_out.strip()):
                        tests_passed+=1
                    else:
                        if stderr_data.decode()=='':
                            if files[i] not in runtime_failed:
                                test_failed.append(files[i])
                                test_failed_details.append('Test Index: '+str(j)+' Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout))
                                
                        else:
                            if files[i] not in test_failed:
                                runtime_failed.append(files[i])
                                runtime_failed_details.append('Test Index: '+str(j)+' Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode()))
                                

            except Exception as e:
                print(e)
                compile_failed.append(files[i])


            # try:
            #     print('Filename: ', files[i])
            #     subprocess.run("javac "+translation_dir+"/"+ files[i], check=True, capture_output=True, shell=True, timeout=30)

            #     with open(test_dir+"/"+ files[i].split(".")[0]+"_in.txt" , 'r') as f:
            #         f_in = f.read()
                
            #     if args.test_type == "misleading_test":
            #         f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_misleading_out.txt", "r").read()
            #     else:
            #         f_out = open(test_dir+"/"+ files[i].split(".")[0]+"_out.txt", "r").read()
            #     p = Popen(['java', files[i].split(".")[0]], cwd=translation_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)    

            #     try:
            #         stdout, stderr_data = p.communicate(input=f_in.encode(), timeout=100)
            #     except subprocess.TimeoutExpired:
            #         infinite_loop.append(files[i])
            #         continue
                
            #     print(stdout.decode().strip())
            #     print(f_out.strip())
            #     exit(0)
            #     if(stdout.decode().strip()==f_out.strip()): # stdout is from the translated code , f_out test data from original language 
            #         test_passed.append(files[i])
            #     else:
            #         if stderr_data.decode()=='':
            #             test_failed.append(files[i])
            #             test_failed_details.append('Filename: '+files[i]+' Actual: '+str(f_out)+' Generated: '+ str(stdout.decode()))  
            #         else:
            #             runtime_failed.append(files[i])
            #             runtime_failed_details.append('Filename: '+ files[i]+' Error_type: '+ str(stderr_data.decode())) 
            
            # except Exception as e:
            #     print(e)
            #     compile_failed.append(files[i])

        #remove all .class files generated
        dir_files = os.listdir(translation_dir)
        for fil in dir_files:
            if ".class" in fil: os.remove(translation_dir +"/"+ fil)

    else:
        print("language:{} is not yet supported. select from the following languages[Python,Java]".format(args.target_lang))
        return
    log_path = translation_dir + '/pass_id.txt'
    print(log_path)
    with open(log_path, 'w', encoding='utf-8') as wr:
        for i in test_passed:
            wr.write(i + '\n')
    test_failed = list(set(test_failed))
    test_failed_details = list(set(test_failed_details))
    runtime_failed = list(set(runtime_failed))
    runtime_failed_details = list(set(runtime_failed_details))
    compile_failed = list(set(compile_failed))
    infinite_loop = list(set(infinite_loop))
    test_passed = list(set(test_passed))
    print("#Success:", len(test_passed))
    print("Success:", test_passed)
    print("Fail:", test_failed + runtime_failed + compile_failed + infinite_loop)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='execute codenet tests')
    parser.add_argument('--dataset', help='dataset to use for code translation. should be one of [codenet,avatar]', required=True, type=str)
    parser.add_argument('--source_lang', help='source language to use for code translation. should be one of [python,java]', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation. should be one of [python,java]', required=True, type=str)
    parser.add_argument('--model', help='model to use for code translation.', required=True, type=str)
    # parser.add_argument('--report_dir', help='path to directory to store report', required=True, type=str)
    parser.add_argument('--test_type', help='test_type', required=True, type=str)
    parser.add_argument('--der', action='store_true')
    args = parser.parse_args()

    main(args)
