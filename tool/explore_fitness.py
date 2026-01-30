import csv
import os
import sys
import json
import numpy as np

complexity_consistency = ["changing_AugAssign", "change_var_names", "change_function_names", "move_functions_into_new_class"]

def collect_complexity(complexity_dir):
    instance_results = {}
    for dirpath, _, files in os.walk(complexity_dir):
        for file in files:
            if file.endswith(".csv") and "readability" not in file and "complexity" not in file and "correlation" not in file:
                file_path = os.path.join(dirpath, file)
                with open(file_path, mode ='r')as file:
                    csvFile = csv.reader(file)
                    for line in csvFile:
                        if "rule" in line:
                            continue
                        try:
                            rule = line[0]
                            instance = line[1]
                            complexity_diff = line[-1]
                            if instance not in instance_results:
                                instance_results[instance] = {}
                            if rule not in instance_results[instance] and rule not in complexity_consistency:
                                instance_results[instance][rule] = float(complexity_diff)
                        except:
                            pass
                        
    for instance in instance_results:
        instance_results[instance] = dict(sorted(instance_results[instance].items(), key=lambda item: item[1], reverse=True))
        # print(instance, instance_results[instance])
    return instance_results

def collect_readability(readability_dir):
    instance_results = {}
    for dirpath, _, files in os.walk(readability_dir):
        for file in files:
            if file.endswith("_readability.csv"):
                rule = file.split("_readability")[0]
                file_path = os.path.join(dirpath, file)
                with open(file_path, mode ='r')as file:
                    csvFile = csv.reader(file)
                    for line in csvFile:
                        if "File1" in line:
                            continue
                        try:
                            instance = line[0].split("/")[-1]
                            token_LD = float(line[-1])
                            if instance not in instance_results:
                                instance_results[instance] = {}
                            if rule not in instance_results[instance] and rule not in complexity_consistency:
                                instance_results[instance][rule] = token_LD
                        except:
                            pass
                        
    for instance in instance_results:
        if instance_results[instance] != {}:
            instance_results[instance] = dict(sorted(instance_results[instance].items(), key=lambda item: item[1]))    
            # print(instance, instance_results[instance])
    return instance_results

def remove_unmatched_keys(dict1, dict2):
    unique_to_dict1 = set(dict1.keys()) - set(dict2.keys())
    unique_to_dict2 = set(dict2.keys()) - set(dict1.keys())

    new_dict1 = {k: v for k, v in dict1.items() if k not in unique_to_dict1}
    new_dict2 = {k: v for k, v in dict2.items() if k not in unique_to_dict2}

    return new_dict1, new_dict2

def combine_complexity_readability(complexity_dict, readability_dict, num_candidates_to_select):
    final_result = {}
    for instance in complexity_dict:
        if instance in readability_dict:
            if len(complexity_dict[instance]) == 0 or len(readability_dict[instance]) == 0:
                continue
            if len(complexity_dict[instance]) != len(readability_dict[instance]):
                complexity_dict[instance], readability_dict[instance] = remove_unmatched_keys(complexity_dict[instance], readability_dict[instance])
            final_result[instance] = {
                "complexity": complexity_dict[instance],
                "readability": readability_dict[instance]
            }
    for instance in final_result:
        ranked_data = {}
        fitness = {}
        for key, sub_dict in final_result[instance].items():
            if key == 'readability':
                ranked_data[key] = rank_values_by_ranking(sub_dict, reverse=True)
            elif key == "complexity":
                ranked_data[key] = rank_values_by_ranking(sub_dict, reverse=False)
        for rule in ranked_data["complexity"]:
            fitness[rule] = round(0.5 * ranked_data["complexity"][rule] + 0.5 * ranked_data["readability"][rule], 2)
        final_result[instance]["fitness"] = dict(sorted(fitness.items(), key=lambda item: item[1], reverse=True))
        final_result[instance]["rank"] = ranked_data
        # print(ranked_data) 
        pareto_optimal_set = get_pareto_optimal_set(ranked_data)
        # print("old:", pareto_optimal_set)
        candidates = list(ranked_data["complexity"].keys())
        complexity_values = np.array([ranked_data["complexity"][c] for c in candidates])
        readability_values = np.array([ranked_data["readability"][c] for c in candidates])
        values = np.column_stack((complexity_values, readability_values))
        pareto_fronts = non_dominated_sorting(values)

        # Pareto rank and crowding distance
        selected_candidates = []
        for front in pareto_fronts:
            if len(selected_candidates) + len(front) > num_candidates_to_select:
                distances = crowding_distance(values, front)
                sorted_front = [front[i] for i in np.argsort(-distances)]
                selected_candidates.extend(sorted_front[:num_candidates_to_select - len(selected_candidates)])
                break
            else:
                selected_candidates.extend(front)
        selected_candidates_names = [candidates[i] for i in selected_candidates]
        # print("Selected Candidates:", selected_candidates_names)
        final_result[instance]["pareto_optimal_set"] = selected_candidates_names
    
    return final_result

def non_dominated_sorting(values):
    pareto_fronts = []
    num_solutions = values.shape[0]
    domination_counts = np.zeros(num_solutions)
    dominated_solutions = [[] for _ in range(num_solutions)]
    
    for i in range(num_solutions):
        for j in range(num_solutions):
            if np.all(values[i] <= values[j]) and np.any(values[i] < values[j]):
                dominated_solutions[i].append(j)
            elif np.all(values[j] <= values[i]) and np.any(values[j] < values[i]):
                domination_counts[i] += 1
                
        if domination_counts[i] == 0:
            pareto_fronts.append([i])
    
    current_front = 0
    while len(pareto_fronts[current_front]) > 0:
        next_front = []
        for i in pareto_fronts[current_front]:
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        current_front += 1
        pareto_fronts.append(next_front)
    
    return pareto_fronts[:-1]

def crowding_distance(values, pareto_front):
    distances = np.zeros(len(pareto_front))
    for m in range(values.shape[1]):
        sorted_indices = np.argsort(values[pareto_front, m])
        distances[sorted_indices[0]] = distances[sorted_indices[-1]] = np.inf
        for k in range(1, len(pareto_front) - 1):
            distances[sorted_indices[k]] += (
                values[pareto_front, m][sorted_indices[k + 1]] - 
                values[pareto_front, m][sorted_indices[k - 1]]
            )
    return distances
        
def get_pareto_optimal_set(rank):
    criteria = list(rank.keys())
    
    items = {}
    for criterion in criteria:
        for item, rank_value in rank[criterion].items():
            if item not in items:
                items[item] = {}
            items[item][criterion] = rank_value

    def is_dominated(item1, item2):
        dominated = False
        for criterion in criteria:
            if items[item1][criterion] > items[item2][criterion]:
                return False
            elif items[item1][criterion] < items[item2][criterion]:
                dominated = True
        return dominated

    pareto_optimal_set = set(items.keys())
    for item1 in items:
        for item2 in items:
            if item1 != item2 and is_dominated(item1, item2):
                pareto_optimal_set.discard(item1)
                break

    return list(pareto_optimal_set)
        
def rank_values_by_ranking(sub_dict, reverse=False):
    sorted_items = sorted(sub_dict.items(), key=lambda item: item[1], reverse=reverse)
    # ranks = {item[0]: rank+1 for rank, item in enumerate(sorted_items)}
    ranks = {}
    current_rank = 1
    last_value = None
    
    for item in sorted_items:
        key, value = item
        if value != last_value:
            rank = current_rank
            last_value = value
        ranks[key] = rank
        current_rank += 1

    norm_ranks = {}
    for key in ranks:
        norm_ranks[key] = round(ranks[key]/len(ranks),3)
    return norm_ranks

def rank_values(sub_dict, reverse=False):
    sorted_items = sorted(sub_dict.items(), key=lambda item: item[1], reverse=reverse)
    # ranks = {item[0]: rank+1 for rank, item in enumerate(sorted_items)}
    ranks = {}
    current_rank = 1
    last_value = None
    
    for item in sorted_items:
        key, value = item
        if value != last_value:
            rank = current_rank
            last_value = value
        ranks[key] = rank
        current_rank += 1

    norm_ranks = {}
    for key in ranks:
        norm_ranks[key] = ranks[key]
    return norm_ranks

def main(complexity_dir, readability_dir, num_candidates_to_select):
    complexity_dict = collect_complexity(complexity_dir)
    readability_dict = collect_readability(readability_dir)
    final_result = combine_complexity_readability(complexity_dict, readability_dict, num_candidates_to_select)
    count_ops = {}
    for instance in final_result:
        # print(instance, final_result[instance])
        for op in final_result[instance]["pareto_optimal_set"]:
            if op not in count_ops:
                count_ops[op] = 0
            count_ops[op] += 1
    # print(len(final_result))
    sorted_count_ops = dict(sorted(count_ops.items(), key=lambda item: item[1], reverse=True))
    for op in sorted_count_ops:
        print(op, sorted_count_ops[op])
    
    with open("sample2.json", "w") as outfile: 
        json.dump(final_result, outfile)

if __name__ == "__main__":
    args = sys.argv[1:]
    complexity_dir = args[0]
    readability_dir = args[1]
    num_candidates_to_select = 5
    main(complexity_dir, readability_dir, num_candidates_to_select)
    
    """#python3 explore_fitness.py /home/yang/Benchmark/v2_metric_avatar_transformation/44bbf303f10d2e9ddbdd85c1b4e071
f52523f182/summary /home/yang/Benchmark/v2_metric_avatar_transformation/44bbf303f10d2e9ddbdd85c1b4e071f52523f182/CSV > sample1.log
    """