import os
import logging
from dotenv import load_dotenv
import argparse
from matplotlib.pylab import dtype
import torch
import openai
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from vllm import LLM, SamplingParams
import openai
import json



os.makedirs(f'logs', exist_ok=True)
logging.basicConfig(filename=f"logs/translation.log", level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
    
def main(args):
    alter_avatar_examples = read_json('avatar_alter.json')

    extensions = {'Python': 'py', 'Java': 'java'}

    experiment_type = 'no_test'
    if args.use_misleading_test:
        experiment_type = 'misleading_test'
    elif args.use_test:
        experiment_type = 'use_test'
    
    test_type = 'tests'
    if args.use_misleading_test:
        test_type = 'misleading_tests'

    in_folder = f'../dataset/Intermediate/Translation/Avatar/Python/code'
    test_folder = f'../dataset/Intermediate/Translation/Avatar/Python/tests'
    if args.der:
        out_folder = f'../Experiment_Results/intermediate/DER/Translation/{experiment_type}/{args.model.split("/")[-1]}/{args.dataset}/{args.source_lang}/{args.target_lang}'
    else:
        out_folder = f'../Experiment_Results/intermediate/SR/Translation/{experiment_type}/{args.model.split("/")[-1]}/{args.dataset}/{args.source_lang}/{args.target_lang}'

    in_files = os.listdir(in_folder)
    print(f'found {len(in_files)} inputs')

    # check for files alraedy extracted
    already_extracted_files = []
    if os.path.exists(out_folder):
        already_extracted_files = os.listdir(out_folder)
        if len(already_extracted_files) > 0:
            already_extracted_files = [x.split('.')[0] for x in already_extracted_files if os.stat(f'{out_folder}/{x}').st_size != 0]

    if len(already_extracted_files) > 0:
        in_files = [x for x in in_files if x.split('.')[0] not in already_extracted_files]

    ext = extensions[args.target_lang]
    device = 'cuda:0,1'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"   # pick GPU 0 and 1

    tokenizer, model = None, None
    if 'gpt-' in args.model :
        client = openai.OpenAI(
                    # This is the default and can be omitted
                    api_key=os.getenv('OPENAIKEY'),
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            token=os.getenv('AUTH_TOKEN'),
            cache_dir=args.cache_dir
        )
        vllm_model = LLM(
            model=args.model,
            trust_remote_code=True,
            max_model_len=4096,
            tensor_parallel_size=2,
            dtype="bfloat16",
            # device_map=device
        )
        sampling = SamplingParams(
            max_tokens=4096,
            temperature=0.0,
        )
    prompts = []
    file_names = []

    os.makedirs(out_folder, exist_ok=True)
    print(f"Translating {len(in_files)} files...")
    for f in tqdm(in_files):
        if "pycache" in f:
            continue
        if "atcoder" not in f and "codeforces" not in f:
            continue
        base_name = f.split('.')[0]
        test_input = open(f'{test_folder}/{base_name}_0.in', 'r').read()
        test_output = ''
        if args.use_test:
            test_output = open(f'{test_folder}/{base_name}_0.out', 'r').read()

        if len(test_input) > 500 or len(test_output) > 500:
            # continue
            test_input = test_input[:500]
            test_output = test_output[:500]

        prompt_file = f'{in_folder}/{f}'
        icl = ""
        if base_name in alter_avatar_examples:
            icl = alter_avatar_examples[base_name]

        with open(prompt_file, "r", encoding="ISO-8859-1", errors='ignore') as fin:
            prompt = fin.readlines()
            # add ICL
            prompt += f"\n```\n\nThe following is a semantically equivalent program which may help your understanding:\n```{icl}\n"

            if 'codellama' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = f"Translate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n"
                else:
                    prompt = f"Translate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```:\n```\n" + "".join(prompt) + "\n```\n. Do not print other explanation."
                prompt = f"[INST] <<SYS>>\nYou are an expert {args.target_lang} programmer and assistant\n<</SYS>>\n\n{prompt}[/INST]\n"

            elif 'magicoder' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\nTranslate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n\n@@ Response\n"
                else:
                    prompt = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\nTranslate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```:\n```\n" + "".join(prompt) + "\n```\n\n@@ Response\n"

            elif 'wizardcoder' in args.model or 'WizardCoder' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nTranslate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n\n### Response:\n"
                else:
                    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nTranslate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```:\n```\n" + "".join(prompt) + "\n```\n\nDo not print other explanation.\n### Response:\n"

            elif 'deepseek-coder' in args.model or 'semcoder' in args.model or 'multi' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = f"You are an expert Python programmer.Your task is to write a Python function to solve a programming problem.\n\n### Instruction:\nTranslate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n\n### Response:\n"
                else:
                    prompt = f"You are an expert Python programmer.Your task is to write a Python function to solve a programming problem.\n\n### Instruction:\nTranslate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```:\n```\n" + "".join(prompt) + "\n```\n\nDo not print other explanation.\n### Response:\n"

            elif 'Mistral' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = f"[INST] Translate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n[/INST]\n"
                else:
                    prompt = f"[INST] Translate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```:\n```\n" + "".join(prompt) + "\n```\n[/INST]\n"

            elif 'starcoder' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = f"Translate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + f"\n```\n\n{args.target_lang} code:"
                else:
                    prompt = f"Translate the following {args.source_lang} code to {args.target_lang} and enclose your solution inside ```{args.target_lang.lower()} and ```:\n```\n" + "".join(prompt) + f"\n```\n\n{args.target_lang}.Do not print other explanation.\n code:"
                prefix_token = "<fim_prefix>"
                suffix_token = "<fim_suffix><fim_middle>"
                prompt = prefix_token + prompt + suffix_token
            

            elif 'gpt' in args.model: #in ['gpt-4', 'gpt-3.5', 'gpt-4-turbo']:
                if args.use_test or args.use_misleading_test:
                    prompt = "Translate the following code from " + args.source_lang + " to " + args.target_lang + " and enclose your solution inside ```" + args.target_lang.lower() + "and" + "```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n"
                else:
                    prompt = "Translate the following code from " + args.source_lang + " to " + args.target_lang + " and enclose your solution inside ```" + args.target_lang.lower() + "and" + "```:\n```\n" + "".join(prompt) + "\n```\n"

            elif 'CodeQwen' in args.model:
                if args.use_test or args.use_misleading_test:
                    prompt = "Translate the following code from " + args.source_lang + " to " + args.target_lang + " and enclose your solution inside ```" + args.target_lang.lower() + "```.\nA sample test case is provided below:\n\nTest input:\n" + test_input + "\nExpected output:\n" + test_output + "\n\n```\n" + "".join(prompt) + "\n```\n"
                else:
                    prompt = "Translate the following code from " + args.source_lang + " to " + args.target_lang + " and enclose your solution inside ```" + args.target_lang.lower() + "```:\n```\n" + "".join(prompt) + "\n```\n"
            prompts.append(prompt)
            file_names.append(f)
    
    # ---- generate AFTER the loop ----
    if not prompts:
        return  # nothing to do

    if 'gpt-' in args.model:
        # OpenAI Chat API – one by one (Chat API is per request)
        for f, prompt in zip(file_names, prompts):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                generated_output = resp.choices[0].message.content
                out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'
                with open(out_file, 'w') as fot:
                    print(generated_output, file=fot)
            except Exception as e:
                print("Error:", e)
    else:
        # vLLM – batch generate in parallel
        vllm_outputs = vllm_model.generate(prompts, sampling)
        for f, out in zip(file_names, vllm_outputs):
            generated_output = out.outputs[0].text
            out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'
            with open(out_file, 'w') as fot:
                print(generated_output, file=fot)
    # ---------------------------------


            # try:
            #     if 'gpt' in args.model:
            #         try:
            #                 client = OpenAI(
            #                     # base_url="http://127.0.0.1:11434/v1", #11434 #130.126.137.50
            #                     api_key=os.environ.get('OPENAI_API_KEY'),
            #                 )
            #                 raw_outputs = ''
            #                 print(prompt)
            #                 message = [
            #                     {"role": "user", "content": "You are a helpful assistant."},
            #                     {"role": "user", "content": prompt}]

            #                 raw_outputs = client.chat.completions.create(
            #                     model=args.model,
            #                     messages=message,
            #                     temperature=0,
            #                 )
            #                 generated_output = raw_outputs.choices[0].message.content
            #                 print(generated_output)
            #         except Exception as e:
            #             print(e)
            #             generated_output = e
            #             continue                                                            
            #     else:
            #         tokenizer = AutoTokenizer.from_pretrained(args.model, token=os.getenv('AUTH_TOKEN'), cache_dir=args.cache_dir)
            #         model = LLM(model=args.model, trust_remote_code = True, max_model_len = 4096, tensor_parallel_size = 2, dtype = "bfloat16")
            #         sampling = SamplingParams(
            #             max_tokens=4096,
            #             temperature=0.0,   # deterministic (like do_sample=False)
            #         )

            #         # Generate
            #         outputs = model.generate([prompt], sampling)
            #         generated_output = outputs[0].outputs[0].text
            #         print("Generated output:", generated_output)


            #     out_file = f'{out_folder}/{f.split(".")[0]}.{ext}'
            #     with open(out_file, 'w') as fot:
            #         print(generated_output, file=fot)

            # except (ValueError, FileNotFoundError) as e:
            #     print(e)
            #     continue

if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description='run translation with open-source models given dataset and languages')
    parser.add_argument('--model', help='model to use for code translation', required=True, type=str)
    parser.add_argument('--dataset', help='dataset to use for code translation', required=True, type=str)
    parser.add_argument('--source_lang', help='source language to use for code translation', required=True, type=str)
    parser.add_argument('--target_lang', help='target language to use for code translation', required=True, type=str)
    parser.add_argument('--gpu_id', help='gpu id to use', default=0, type=int)
    parser.add_argument('--cache_dir', help='cache directory for huggingface models', required=False, type=str)
    parser.add_argument('--use_test', help='use test dataset', action='store_true')
    parser.add_argument('--use_misleading_test', help='use test dataset', action='store_true')
    parser.add_argument('--der', action='store_true')
    args = parser.parse_args()

    # Initialize configurations
    source = args.source_lang
    target = args.target_lang
    logging.info(f"translating examples from {source} to {target} using {args.model} and {args.dataset} dataset")
    main(args)
