# GeneBench

## Repository Structure

⭐️ Transformations of original benchmarks that constructed the final problems in GeneBench ([transformations](transformations)).  

⭐️ Detailed logs of the transformation process for each individual program, including statistics for each transformation across all iterations, such as time cost, applied operators, and readability/complexity metrics; GeneBench benchmark, Pareto-front solutions, and non-optimal solutions: [all_logs/GA_logs.zip](https://drive.google.com/drive/folders/1npU--eZxlXLBlb_UOrqoakrb9Io_BL1I?usp=share_link).

⭐️ Scripts to reproduce and visualize evaluation results in the paper ([reproduction](reproduction)).  

⭐️ Source code of GeneBench ([tool](tool/)). You can see the implementation of all operators under [tool/operators](tool/operators).   

⭐️ Evaluation scripts ([eval](eval/)). 

⭐️ Details of 22 semantically equivalent transformation operators, along with code examples illustrating how they transform the code: [link](https://drive.google.com/file/d/1clxGcZ4fivTVM7-9hFkTMkly5ZsXqGt1/view?usp=sharing) (three pages).   

⭐️ Few-shot [examples](few-shot.md).


## 

## Usage

**0. Before starting**
- GeneBench works with `Python3.10.12` on `Linux`
- You can run the following steps to set up the environment
```
git clone https://github.com/genebenchartifact-dot/GeneBench
cd GeneBench
bsah -x bash_scripts/setup.sh
python3 -m venv genebench_venv
source genebench_venv/bin/activate
pip3 install -r requirements.txt
```

**1. Run with GeneBench**  

GeneBench supports multiple modes for code transformation:

- You can run the following commands to apply transformations on a single file with any single operator  
```
python3 -u tool/init.py --src_file [input_file] --target_file [output_file] --file_id [file_id] --evaluation_tests_dir [test_directory] -rule [operator]
```
e.g., 
```
python3 -u tool/init.py --src_file dataset/HumanEval/code/HumanEval_0.py --target_file reproduction_examples/HumanEval0_transformation.py --file_id HumanEval_0 --evaluation_tests_dir dataset/HumanEval -rule add_nested_if |& tee reproduction_examples/HumanEval_0_add_nested_if.log
```

- You can run the following commands to apply transformations on a single file with all applicable operators  
```
python3 -u tool/init.py --src_file [input_file] --target_file [output_file] --file_id [file_id] --evaluation_tests_dir [test_directory]
```
e.g., 
```
python3 -u tool/init.py --src_file dataset/HumanEval/code/HumanEval_0.py --target_file reproduction_examples/HumanEval0_transformation.py --file_id HumanEval_0 --evaluation_tests_dir dataset/HumanEval |& tee reproduction_examples/HumanEval_0.log
```

- You can run the following commands to apply the genetic algorithm across all benchmarks. The maximum number of parallel processes can be configured via ${max_session}, which will then be executed automatically in parallel tmuxes.

```
bash -x bash_scripts/bash_scripts/call_ga.sh ${max_session}
```

To check the results, each instance will have a seperate directory, with the following outputs:
```
- ga/generations: all transformations during the whole genetic algorithm process
- ga/jsons: details and logs of all generations
- ga/selection: final selected transformation (you can also view it at [instance].patch)
- [instance].log: all logs for each instance
```

