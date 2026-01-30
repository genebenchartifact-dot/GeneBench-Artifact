import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import csv
from scipy.stats import ttest_ind, mannwhitneyu
from collections import defaultdict
from typing import List, Dict
from collections import Counter


def remove_overlap(list1: List, list2: List):
    set1, set2 = set(list1), set(list2)
    unique_list1 = [item for item in list1 if item not in set2]
    unique_list2 = [item for item in list2 if item not in set1]
    return unique_list1, unique_list2
    
def filter_frequent_elements(lst: List, threshold: int = 3):
    counts = Counter(lst)
    filtered_list = [item for item in lst if counts[item] > threshold]
    return filtered_list

def count_occurrences(lst: List):
    return dict(Counter(lst))

def cal_increase_times(original_rc, transformation_rc):
    times = (transformation_rc - original_rc)/original_rc
    return times

def cal_relative_complexity(complexity):
    relative_complexity = 0
    for key in complexity:
        if key in ["File", "Log"]:
            continue
        diff = float(complexity[key])/float(target_complexity[key])
        relative_complexity += diff
    return relative_complexity/7

def compare3(true_true_list, true_false_list):
    print(f"true_true_list: {len(true_true_list)}")
    print(f"true_false_list: {len(true_false_list)}")
    
    stats_true_true = {
        "Mean": np.mean(true_true_list),
        "Median": np.median(true_true_list),
        "Std Dev": np.std(true_true_list),
        "Min": np.min(true_true_list),
        "Max": np.max(true_true_list)
    }
    
    stats_true_false = {
        "Mean": np.mean(true_false_list),
        "Median": np.median(true_false_list),
        "Std Dev": np.std(true_false_list),
        "Min": np.min(true_false_list),
        "Max": np.max(true_false_list)
    }

    t_stat, p_val_ttest = stats.ttest_ind(true_true_list, true_false_list, equal_var=False)
    u_stat, p_val_mannwhitney = stats.mannwhitneyu(true_true_list, true_false_list, alternative='two-sided')

    print("Descriptive Statistics:")
    print(f"true_true_similarity: {stats_true_true}")
    print(f"true_false_similarity: {stats_true_false}")
    print("\nMann-Whitney U test:")
    print(f"U-statistic = {u_stat:.4f}, p-value = {p_val_mannwhitney:.7f}")
    print("\nWelch's T-test:")
    print(f"t-statistic = {t_stat:.4f}, p-value = {p_val_ttest:.7f}")

    data = {
        "Similarity Score": true_true_list + true_false_list,
        "Category": [r"Success_Success$^3$"] * len(true_true_list) + [r"Success_Failure$^3$"] * len(true_false_list)
    }

    plt.figure(figsize=(8.5, 3.5))
    sns.violinplot(x=data["Category"], y=data["Similarity Score"], palette=["#b5ea8c", "pink"], inner="quartile")
    plt.text(0, stats_true_true["Mean"], f"Mean: {stats_true_true['Mean']:.2f}\nMedian: {stats_true_true['Median']:.2f}", #\nMedian: {stats_true_true['Median']:.2f}
             ha='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='black', alpha=0.6))
    
    plt.text(1, stats_true_false["Mean"], f"Mean: {stats_true_false['Mean']:.2f}\nMedian: {stats_true_false['Median']:.2f}", #\nMedian: {stats_true_false['Median']:.2f}
             ha='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='black', alpha=0.6))

    plt.text(0.5, max(stats_true_true["Max"], stats_true_false["Max"]) - 0.03, 
             f"Mann-Whitney U-Test\np-value: {p_val_mannwhitney:.8f}",
             ha='center', fontsize=15, color='black') #fontweight='bold'

    plt.ylabel("Relative Complexity", fontsize=18) #fontweight='bold'
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def read_json(json_file):
    with open(json_file, "r") as file:
        return json.load(file)

def read_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line))
    return data



transformation_complexity = {}
original_instance_complexity = {}
target_complexity = {'File': None, 'Base complexity': 46.303630363036305, 'Predicates with operators': 18.254125412541253, 'Nested levels': 17.32013201320132, 'Complex code structures': 15.689768976897689, 'Third-Party calls': 17.056105610561055, 'Inter_dependencies': 4.481848184818482, 'Intra_dependencies': 10.05940594059406, 'Log': None}

def get_original_instance_complexity(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row['File']
            ID = file_path.split('.py')[0].split('/')[-1]
            # relative_complexity = cal_relative_complexity(row)
            if ID not in original_instance_complexity:
                original_instance_complexity[ID] = row
    
def get_transformation_instance_complexity(file_path):
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            file_path = row['File']
            ID = file_path.split('.py')[0].split('/')[-1]
            if ID not in transformation_complexity:
                transformation_complexity[ID] = row

classeval_ids={'ClassEval_0': 'AccessGatewayFilter', 'ClassEval_1': 'AreaCalculator', 'ClassEval_2': 'ArgumentParser', 'ClassEval_3': 'ArrangementCalculator', 'ClassEval_4': 'AssessmentSystem', 'ClassEval_5': 'is', 'ClassEval_6': 'AvgPartition', 'ClassEval_7': 'BalancedBrackets', 'ClassEval_8': 'BankAccount', 'ClassEval_9': 'BigNumCalculator', 'ClassEval_10': 'BinaryDataProcessor', 'ClassEval_11': 'BitStatusUtil', 'ClassEval_12': 'BlackjackGame', 'ClassEval_13': 'BookManagement', 'ClassEval_14': 'BookManagementDB', 'ClassEval_15': 'BoyerMooreSearch', 'ClassEval_16': 'Calculator', 'ClassEval_17': 'CalendarUtil', 'ClassEval_18': 'CamelCaseMap', 'ClassEval_19': 'ChandrasekharSieve', 'ClassEval_20': 'Chat', 'ClassEval_21': 'Classroom', 'ClassEval_22': 'ClassRegistrationSystem', 'ClassEval_23': 'CombinationCalculator', 'ClassEval_24': 'ComplexCalculator', 'ClassEval_25': 'CookiesUtil', 'ClassEval_26': 'CSVProcessor', 'ClassEval_27': 'CurrencyConverter', 'ClassEval_28': 'DatabaseProcessor', 'ClassEval_29': 'DataStatistics', 'ClassEval_30': 'DataStatistics2', 'ClassEval_31': 'DataStatistics4', 'ClassEval_32': 'DecryptionUtils', 'ClassEval_33': 'DiscountStrategy', 'ClassEval_34': 'DocFileHandler', 'ClassEval_35': 'EightPuzzle', 'ClassEval_36': 'EmailClient', 'ClassEval_37': 'EncryptionUtils', 'ClassEval_38': 'ExcelProcessor', 'ClassEval_39': 'ExpressionCalculator', 'ClassEval_40': 'FitnessTracker', 'ClassEval_41': 'GomokuGame', 'ClassEval_42': 'Hotel', 'ClassEval_43': 'HRManagementSystem', 'ClassEval_44': 'HtmlUtil', 'ClassEval_45': 'ImageProcessor', 'ClassEval_46': 'Interpolation', 'ClassEval_47': 'IPAddress', 'ClassEval_48': 'IpUtil', 'ClassEval_49': 'JobMarketplace', 'ClassEval_50': 'JSONProcessor', 'ClassEval_51': 'KappaCalculator', 'ClassEval_52': 'Lemmatization', 'ClassEval_53': 'LongestWord', 'ClassEval_54': 'MahjongConnect', 'ClassEval_55': 'Manacher', 'ClassEval_56': 'MetricsCalculator', 'ClassEval_57': 'MetricsCalculator2', 'ClassEval_58': 'MinesweeperGame', 'ClassEval_59': 'MovieBookingSystem', 'ClassEval_60': 'MovieTicketDB', 'ClassEval_61': 'MusicPlayer', 'ClassEval_62': 'NLPDataProcessor', 'ClassEval_63': 'NLPDataProcessor2', 'ClassEval_64': 'NumberConverter', 'ClassEval_65': 'NumberWordFormatter', 'ClassEval_66': 'NumericEntityUnescaper', 'ClassEval_67': 'Order', 'ClassEval_68': 'PageUtil', 'ClassEval_69': 'PDFHandler', 'ClassEval_70': 'PersonRequest', 'ClassEval_71': 'PushBoxGame', 'ClassEval_72': 'RegexUtils', 'ClassEval_73': 'RPGCharacter', 'ClassEval_74': 'Server', 'ClassEval_75': 'ShoppingCart', 'ClassEval_76': 'SignInSystem', 'ClassEval_77': 'Snake', 'ClassEval_78': 'SplitSentence', 'ClassEval_79': 'SQLGenerator', 'ClassEval_80': 'SQLQueryBuilder', 'ClassEval_81': 'Statistics3', 'ClassEval_82': 'StockPortfolioTracker', 'ClassEval_83': 'StudentDatabaseProcessor', 'ClassEval_84': 'TextFileProcessor', 'ClassEval_85': 'Thermostat', 'ClassEval_86': 'TicTacToe', 'ClassEval_87': 'TimeUtils', 'ClassEval_88': 'TriCalculator', 'ClassEval_89': 'TwentyFourPointGame', 'ClassEval_90': 'URLHandler', 'ClassEval_91': 'UrlPath', 'ClassEval_92': 'UserLoginDB', 'ClassEval_93': 'VectorUtil', 'ClassEval_94': 'VendingMachine', 'ClassEval_95': 'Warehouse', 'ClassEval_96': 'WeatherSystem', 'ClassEval_97': 'Words2Numbers', 'ClassEval_98': 'XMLProcessor', 'ClassEval_99': 'ZipFileProcessor'}


def collect_files(file, similarity, true_true_similarity, true_false_similarity,true_true_instances, true_false_instances):
    data = read_json(file)
    similarity_data = read_jsonl(similarity)

    models = {
        "deepseek_coder_6_7b_instruct": "deepseek-coder-6.7b-instruct",
        "CodeLlama_13b_hf": "CodeLlama-13b-hf",
        "deepseek_coder_6_7b_base": "deepseek-coder-6.7b-base",
        "starcoder2_15b": "starcoder2-15b",
        "CodeLlama_13b_Instruct_hf": "CodeLlama-13b-Instruct-hf",
        "deepseek_coder_33b_instruct": "deepseek-coder-33b-instruct",
        "CodeLlama_34b_Instruct_hf": "CodeLlama-34b-hf",
        "WizardCoder_33B_V1_1": "WizardCoder-33B-V1.1",
        "WizardCoder_15B_V1_0": "WizardCoder-15B-V1.0",
        "semcoder": "semcoder_1030",
    }
    for model_key in data:
        if "gpt" in model_key:
            continue
        model = model_key        
        if "avatar" in file:
            model = model_key.replace("-", "_")
            model = models[model]
        elif "cruxeval" in file:
            if 'WizardCoder-Python-34B-V1.0' in model_key:
                model = 'WizardCoder-33B-V1.1'
            else:
                model = model_key.split("_temp")[0].replace("-", "_").replace(".", "_")
                model = models[model]
        elif 'classeval' in file or 'humaneval' in file:
            model = model_key.replace("-", "_").replace(".", "_")
            model = models[model]
        if model not in true_true_similarity:
            true_true_similarity[model] = []
            true_false_similarity[model] = []
            true_true_instances[model] = {}
            true_false_instances[model] = {}

        for item in similarity_data:
            if item['instance'] in data[model_key]['true_true']:
                if item['instance'] not in true_true_instances[model]: # to exlude duplicates
                    true_true_instances[model][item['instance']] = item['embedding similarity'][model]
                    true_true_similarity[model].append(item['embedding similarity'][model])
            if item['instance'] in data[model_key]['true_false']:
                if item['instance'] not in true_false_instances[model]:
                    true_false_instances[model][item['instance']] = item['embedding similarity'][model]
                    true_false_similarity[model].append(item['embedding similarity'][model])

    print(f"Processed {file}\n")


def main():
    true_true_similarity = {}
    true_false_similarity = {}
    true_true_instances = {}
    true_false_instances = {}

    files = [
        ("inputs/true_false_avatar.json",
         "inputs/avatar.jsonl"),
        ("inputs/true_false_classeval.json",
         "inputs/classeval.jsonl"),
        ("inputs/true_false_humaneval.json",
         "inputs/humaneval.jsonl"),
        ("inputs/true_false_cruxeval_id_output.json",
         "inputs/cruxeval.jsonl"),
        ("inputs/true_false_cruxeval_id_input.json",
         "inputs/cruxeval.jsonl")
    ]
    

    for file, similarity in files:
        print(file)
        collect_files(file, similarity, true_true_similarity, true_false_similarity, true_true_instances, true_false_instances)
    
    print("\nFinal Statistics across all models:")
    model_lists = {}
    true_true_instances_across_all = []
    true_false_instances_across_all = []
    
    for model in true_true_instances:
        model_lists[model] = {'true_true': [], 'true_false': []}
        for instance in true_true_instances[model]:
            if instance not in true_false_instances[model]: # to exclude instances that are in both true_true and true_false
                model_lists[model]['true_true'].append(true_true_instances[model][instance])
                # if instance not in true_true_instances_across_all:
                true_true_instances_across_all.append(instance)
        for instance in true_false_instances[model]:
            if instance not in true_true_instances[model]:
                model_lists[model]['true_false'].append(true_false_instances[model][instance])
                # if instance not in true_false_instances_across_all:
                true_false_instances_across_all.append(instance)
    all_models_true = []
    all_models_false = []
    for model in model_lists:
        true_true_list = model_lists[model]['true_true']
        true_false_list = model_lists[model]['true_false']
        all_models_true.extend(true_true_list)
        all_models_false.extend(true_false_list)
        # print(len(model_lists[model]['true_true']), len(model_lists[model]['true_false']))
    
    # compare3(all_models_true, all_models_false)
    # print("lengths of two lists:", len(all_models_true), len(all_models_false))
    
    print('true_true/true_false instances across all models')
    print(f"true_true_instances_across_all: {len(true_true_instances_across_all)}")
    print(f"true_false_instances_across_all: {len(true_false_instances_across_all)}")
    # print(true_false_instances_across_all)
    
    file_paths =[
        'after/avatar_complexity.csv',
        'after/classeval_complexity.csv',
        'after/humaneval_complexity.csv',
        'after/cruxeval_complexity.csv',
        ] 
    for file_path in file_paths:
        get_transformation_instance_complexity(file_path)
        
    original_files = [
        'existing/avatar_complexity.csv',
        'existing/classeval_complexity.csv',
        'existing/cruxeval_complexity.csv',
        'existing/humaneval_complexity.csv',
    ]
    
    for file_path in original_files:
        get_original_instance_complexity(file_path)
    
    true_true_complexity_vector = []
    true_false_complexity_vector = []
    
    true_true_rc = []
    true_false_rc = []
    
    true_true_rc_delta = []
    true_false_rc_delta = []
    
    more_than_3_true_true = filter_frequent_elements(true_true_instances_across_all)
    more_than_3_true_false = filter_frequent_elements(true_false_instances_across_all)
    
    more_than_3_true_true, more_than_3_true_false = remove_overlap(more_than_3_true_true, more_than_3_true_false)
    
    for instance in more_than_3_true_true: #true_true_instances_across_all
        if instance in transformation_complexity:
            complexity = transformation_complexity[instance]
            rc = cal_relative_complexity(complexity)
            if "ClassEval" in instance:
                org_instance = classeval_ids[instance]
                rc_original = cal_relative_complexity(original_instance_complexity[org_instance])
            else:
                rc_original = cal_relative_complexity(original_instance_complexity[instance])
            true_true_complexity_vector.append(complexity)
            true_true_rc.append(rc)
            delta = cal_increase_times(rc_original, rc)
            true_true_rc_delta.append(delta)
    for instance in more_than_3_true_false: #true_false_instances_across_all
        if instance in transformation_complexity:
            complexity = transformation_complexity[instance]
            rc = cal_relative_complexity(complexity)
            if "ClassEval" in instance:
                org_instance = classeval_ids[instance]
                rc_original = cal_relative_complexity(original_instance_complexity[org_instance])
            else:
                rc_original = cal_relative_complexity(original_instance_complexity[instance])
            true_false_complexity_vector.append(complexity)
            true_false_rc.append(rc)
            delta = cal_increase_times(rc_original, rc)
            true_false_rc_delta.append(delta)
    
    print('strict:',len(more_than_3_true_true), len(more_than_3_true_false))
    print(len(count_occurrences(more_than_3_true_true)),len(count_occurrences(more_than_3_true_false)))
    compare3(true_true_rc, true_false_rc)
    # print(more_than_3_true_true)
    exit(0)
    
if __name__ == "__main__":
    main()
