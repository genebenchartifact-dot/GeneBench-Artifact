import ast
import copy
import difflib
import itertools
import json
import os
import random
import string
import signal
import shutil
import time
import statistics
import logging
from typing import Tuple, Optional, List, Dict

import numpy as np

import complexity
import utils

# Operators
import operators.AddBase64
import operators.AddCrypto
import operators.AddDatetime
import operators.AddDateutil
import operators.AddElseToFor
import operators.AddElseToWhile
import operators.AddNestedIf
import operators.AddNestedForInside
import operators.AddNestedForOutside
import operators.AddNestedWhileInside
import operators.AddNestedWhileOutside
import operators.AddNestedList
import operators.AddThread
import operators.AddTime
import operators.AddTryExceptInsideFunctions
import operators.AddTryExceptOutsideFunctions
import operators.ChangeAugment
import operators.ChangeFunctionNames
import operators.ChangeVarNames
import operators.CreateFunctions
import operators.CreateModuleDependencies
import operators.LocalVarToGlobalVar
import operators.ReplaceWithNumpy
import operators.AddScipy
import operators.AddSklearn
import operators.AddHttp
import operators.IntroduceConfusingVars
import operators.TransformLoopToRecursion
import operators.TransformFunctionToClass
import operators.AddDecorator
import operators.AprChangeArithOp
import operators.AprChangeCompareOp
import operators.AprChangeConditionOp
import operators.AprChangeReturn
import operators.AprChangeType


# === Targets ===
target_readability = {
    "num_tokens": 5038, "lines_of_code": 402, "num_conditions": 45, "num_loops": 12,
    "num_assignments": 89, "num_max_nested_loop": 2, "num_max_nested_if": 4,
    "num_max_conditions_in_if": 3, "max_line_tokens": 31, "num_of_variables": 45,
    "num_of_arrays": 13, "num_of_operators": 78, "num_of_nested_casting": 2, "entropy": 7
}
target_complexity = {
    "Base complexity": 66.29, "Predicates with operators": 25.59, "Nested levels": 20.73,
    "Complex code structures": 30.44, "Third-Party calls": 18.11, "Inter_dependencies": 19.86,
    "Intra_dependencies": 15.22
}


# ---------- helpers ----------
def _instance_dir_from_logger(logger: logging.Logger) -> str:
    """Infer the per-instance result directory from the logger's FileHandler."""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return os.path.dirname(getattr(h, "baseFilename", "") or getattr(h, "stream", "").name)
    # Fallback to CWD if no file handler
    return os.getcwd()


def _safe_apply(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return args[0], args[1]  # (python_code, applicable_rules)


def _wrap_init(mod):
    return lambda code, rules: _safe_apply(mod.init, code, rules)


# ---------- operator ----------
OP_SIMPLE = {
    "add_thread":               _wrap_init(operators.AddThread),
    "transform_localVar_to_globalVar": _wrap_init(operators.LocalVarToGlobalVar),
    "create_functions":         _wrap_init(operators.CreateFunctions),
    "changing_AugAssign":       _wrap_init(operators.ChangeAugment),
    "change_function_names":    _wrap_init(operators.ChangeFunctionNames),
    "change_var_names":         _wrap_init(operators.ChangeVarNames),
    "replace_with_numpy":       _wrap_init(operators.ReplaceWithNumpy),
    "add_datetime":             _wrap_init(operators.AddDatetime),
    "add_time":                 _wrap_init(operators.AddTime),
    "add_crypto":               _wrap_init(operators.AddCrypto),
    "add_sklearn":              _wrap_init(operators.AddSklearn),
    "add_http":                 _wrap_init(operators.AddHttp),
    "add_scipy":                _wrap_init(operators.AddScipy),
    "add_base64":               _wrap_init(operators.AddBase64),
    "add_dateutil":             _wrap_init(operators.AddDateutil),
    "transform_range_to_recursion": _wrap_init(operators.TransformLoopToRecursion),
    "transform_function_to_class":  _wrap_init(operators.TransformFunctionToClass),
    "add_nested_if":            _wrap_init(operators.AddNestedIf),
    "add_nested_for_in":        _wrap_init(operators.AddNestedForInside),
    "add_nested_for_out":       _wrap_init(operators.AddNestedForOutside),
    "add_nested_while_out":     _wrap_init(operators.AddNestedWhileOutside),
    "add_nested_while_in":      _wrap_init(operators.AddNestedWhileInside),
    "add_else_to_for":          _wrap_init(operators.AddElseToFor),
    "add_else_to_while":        _wrap_init(operators.AddElseToWhile),
    "add_nested_list":          _wrap_init(operators.AddNestedList),
    "add_try_except_outside_functions": _wrap_init(operators.AddTryExceptOutsideFunctions),
    "add_try_except_inside_functions":  _wrap_init(operators.AddTryExceptInsideFunctions),
    "add_decorator":            _wrap_init(operators.AddDecorator),
    # APR
    "apr_change_arith_op":      _wrap_init(operators.AprChangeArithOp),
    "apr_change_compare_op":    _wrap_init(operators.AprChangeCompareOp),
    "apr_change_condition_op":  _wrap_init(operators.AprChangeConditionOp),
    "apr_change_type":          _wrap_init(operators.AprChangeType),
    "apr_change_return":        _wrap_init(operators.AprChangeReturn),
}


def _move_functions_into_new_class(code, rules, target_file):
    try:
        return operators.CreateModuleDependencies.init(code, rules, target_file)
    except Exception:
        return code, rules


def _introduce_confusing_vars(code, rules, max_num=1):
    try:
        return operators.IntroduceConfusingVars.init(code, rules, max_num)
    except Exception:
        return code, rules


# ---------- transformation ----------
def transformation_single_rule(source_file: str, rule: str, applicable_rules: List[str], target_file: str):
    python_code = utils.read_file(source_file)
    if rule in OP_SIMPLE:
        update_content, applicable_rules = OP_SIMPLE[rule](python_code, applicable_rules)
    elif rule == "move_functions_into_new_class":
        update_content, applicable_rules = _move_functions_into_new_class(python_code, applicable_rules, target_file)
    elif rule == "introduce_confusing_vars":
        update_content, applicable_rules = _introduce_confusing_vars(python_code, applicable_rules, max_num=1)
    else:
        raise RuntimeError("Operator Not Supported")
    return update_content, applicable_rules


# ---------- rule ----------
def get_applicable_rules_complexity_non_changing(source_file, target_file=None):
    code = utils.read_file(source_file)
    rules = []
    try:
        _, rules = OP_SIMPLE["changing_AugAssign"](code, rules)
        _, rules = OP_SIMPLE["change_function_names"](code, rules)
        _, rules = OP_SIMPLE["change_var_names"](code, rules)
    except Exception:
        pass
    random_seed = np.random.randint(0, 2**32 - 1)
    random.seed(random_seed)
    return (random.choice(rules) if rules else None), random_seed


def get_applicable_apr(source_file, target_file=None):
    code = utils.read_file(source_file)
    rules = []
    try:
        for k in [
            "apr_change_arith_op", "apr_change_compare_op", "apr_change_condition_op",
            "apr_change_type", "apr_change_return", "changing_AugAssign",
            "change_function_names", "change_var_names"
        ]:
            _, rules = (OP_SIMPLE[k] if k in OP_SIMPLE else (lambda c, r: (c, r)))(code, rules)
        _, rules = _introduce_confusing_vars(code, rules, max_num=1)
    except Exception:
        pass
    random_seed = np.random.randint(0, 2**32 - 1)
    random.seed(random_seed)
    return (random.choice(rules) if rules else None), random_seed


def get_all_apr_rules(source_file, target_file=None):
    code = utils.read_file(source_file)
    rules = []
    for k in ["apr_change_arith_op", "apr_change_compare_op", "apr_change_condition_op", "apr_change_return", "apr_change_type"]:
        _, rules = OP_SIMPLE[k](code, rules)
    return rules


def get_applicable_rules(source_file, target_file=None):
    code = utils.read_file(source_file)
    rules = []
    try:
        for k in [
            "add_nested_for_out", "add_nested_while_out", "add_nested_if",
            "create_functions", "add_try_except_inside_functions",
            "add_else_to_for", "add_else_to_while", "add_nested_list",
            "transform_range_to_recursion", "transform_function_to_class",
            "add_thread", "add_decorator", "replace_with_numpy",
            "add_datetime", "add_time",
            "add_crypto", "add_sklearn", "add_http", "add_scipy", "add_base64", "add_dateutil"
        ]:
            _, rules = OP_SIMPLE[k](code, rules)
        _, rules = _move_functions_into_new_class(code, rules, target_file)
    except Exception:
        pass
    return rules


# ---------- metrics & GA utilities ----------
def get_relative_metrics(individuals):
    updated = utils.get_relative_readability(individuals)
    return utils.get_relative_complexity(updated)


def generate_new_node(variant_file, applicable_rules, applied_rules, complexity_metrics, readability_metrics,
                      relevant_distance_readability, relevant_distance_complexity):
    return {
        "variant_file": variant_file,
        "applicable_rules": applicable_rules,
        "applied_rules": applied_rules,
        "complexity": complexity_metrics,
        "readability": readability_metrics,
        "relevant_distance_readability": relevant_distance_readability,
        "relevant_distance_complexity": relevant_distance_complexity,
    }


def filter_individuals(new_individuals):
    return [new_individuals[i] for i in new_individuals if new_individuals[i]["relative_readability"] >= 0]


def apply_multiple_rules(source_file, target_file, all_rules):
    code_path = source_file
    for rule in all_rules:
        update_content, _ = transformation_single_rule(code_path, rule, [], target_file)
        utils.write_file(target_file, update_content)
        code_path = target_file
    return target_file


def non_dominated_sorting(points):
    num_points = len(points)
    domination_counts = np.zeros(num_points, dtype=int)
    dominated_points = [[] for _ in range(num_points)]
    fronts = [[]]
    for i in range(num_points):
        for j in range(num_points):
            if all(points[i] <= points[j]) and any(points[i] < points[j]):
                dominated_points[j].append(i)
                domination_counts[i] += 1
            elif all(points[j] <= points[i]) and any(points[j] < points[i]):
                dominated_points[i].append(j)
                domination_counts[j] += 1
    current_front = [i for i in range(num_points) if domination_counts[i] == 0]
    fronts.append(current_front)
    while current_front:
        next_front = []
        for i in current_front:
            for j in dominated_points[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        fronts.append(next_front)
        current_front = next_front
    return fronts[:-1]


def crowding_distance(points):
    n = len(points)
    if n == 0:
        return []
    distances = np.zeros(n)
    sorted_indices = np.argsort(points, axis=0)
    distances[sorted_indices[0, :]] = np.inf
    distances[sorted_indices[-1, :]] = np.inf
    for i in range(1, n - 1):
        for j in range(points.shape[1]):
            denom = (points[sorted_indices[-1, j], j] - points[sorted_indices[0, j], j]) or 1.0
            distances[sorted_indices[i, j]] += (points[sorted_indices[i + 1, j], j] - points[sorted_indices[i - 1, j], j]) / denom
    return distances


def get_statistics(final_set):
    if not final_set:
        return {k: 0 for k in ["avg_complexity","avg_readability","min_complexity","min_readability","max_complexity","max_readability"]}
    try:
        cvals = [i['relevant_distance_complexity'] for i in final_set]
        rvals = [i['relevant_distance_readability'] for i in final_set]
        return {
            "avg_complexity": sum(cvals) / len(cvals),
            "avg_readability": sum(rvals) / len(rvals),
            "min_complexity": min(cvals),
            "min_readability": min(rvals),
            "max_complexity": max(cvals),
            "max_readability": max(rvals),
        }
    except Exception:
        return {k: 0 for k in ["avg_complexity","avg_readability","min_complexity","min_readability","max_complexity","max_readability"]}


def get_pareto_optimal_set(candidates, fraction=0.2):
    cvals, rvals, names = [], [], []
    for name in candidates:
        cvals.append(candidates[name]["relevant_distance_complexity"])
        rvals.append(candidates[name]["relevant_distance_readability"])
        names.append(name)
    values = np.column_stack((cvals, rvals))
    fronts = non_dominated_sorting(values)
    sel = []
    quota = round(len(candidates) * fraction)
    for front in fronts:
        if len(sel) >= quota:
            break
        if len(front) > 1:
            distances = crowding_distance(values[front])
            front = [front[i] for i in np.argsort(-distances)]
        sel.extend([names[i] for i in front[:quota - len(sel)]])
    return [candidates[n] for n in sel]


def get_final_pareto_optimal_set(candidates, fraction=0.2):
    complexity_median = statistics.median([candidates[n]["relevant_distance_complexity"] for n in candidates]) if candidates else 0
    names, cvals, rvals = [], [], []
    for name in candidates:
        cvals.append(candidates[name]["relevant_distance_complexity"])
        rvals.append(candidates[name]["relevant_distance_readability"])
        names.append(name)
    values = np.column_stack((cvals, rvals)) if names else np.empty((0, 2))
    fronts = non_dominated_sorting(values) if len(values) else []
    sel = []
    if len(candidates) == 0:
        return []
    if round(len(candidates) * fraction) <= 1:
        quota = 1
    else:
        quota = min(3, round(len(candidates) * fraction)) if candidates else 0
    for front in fronts:
        if len(sel) >= quota:
            break
        if len(front) >= 1:
            sorted_indices = sorted(front, key=lambda i: candidates[names[i]]["relevant_distance_complexity"], reverse=True)
            for i in sorted_indices:
                if len(sel) >= quota:
                    break
                item = names[i]
                if candidates[item]["relevant_distance_complexity"] < complexity_median:
                    continue
                sel.append(item)
            if sel:
                break
    return [candidates[n] for n in sel]


def get_pareto_optimal_set_relative(candidates, num_to_select):
    names = list(candidates.keys())
    cvals = np.array([candidates[n]["relevant_distance_complexity"] for n in candidates])
    rvals = np.array([candidates[n]["relevant_distance_readability"] for n in candidates])
    values = np.column_stack((cvals, rvals))
    fronts = non_dominated_sorting(values)
    sel = []
    for front in fronts:
        if len(sel) >= num_to_select:
            break
        if len(front) > 1:
            distances = crowding_distance(values[front])
            front = [front[i] for i in np.argsort(-distances)]
        sel.extend([names[i] for i in front[:num_to_select - len(sel)]])
    return [candidates[n] for n in sel]


# ---------- initialize ----------
def initialize(
    source_file: str,
    single_rule: Optional[str],
    genetic_algorithm: bool,
    target_file: str,
    file_id: Optional[str] = None,
    evaluation_tests_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    time_budget: float = 60.0,
    apr: bool = False,
):
    """
    Unified entry used by init.py.
    - If genetic_algorithm is True: run GA (ga_whole_benchmark) on this single file, pick best variant,
      return (code_string, applied_rules_list).
    - If single_rule is provided: apply that rule and return (code_string, rules_list).
    - Else: apply all-applicable rules sequentially and return (code_string, rules_list).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if genetic_algorithm:
        logger.info("[INIT] Running GA for file_id=%s, time_budget=%.2fs", file_id, time_budget)
        result = _run_ga_and_pick_best([source_file], evaluation_tests_dir, time_budget, apr, logger)
        if result is None:
            logger.warning("[INIT] GA finished but no selection produced; returning None.")
            return None, None
        best_file_path, applied_rules = result
        return utils.read_file(best_file_path), applied_rules

    if single_rule:
        logger.info("[INIT] Applying single rule '%s' to %s", single_rule, source_file)
        updated, rules = transformation_single_rule(source_file, single_rule, [], target_file)
        return updated, rules

    logger.info("[INIT] Applying all-applicable rules sequentially to %s", source_file)
    updated, rules = transformation_all_applicable_rules(source_file, [], target_file, file_id, evaluation_tests_dir, logger)
    return updated, rules


# ---------- test checks ----------
def check_tests(rule, applicable_rules, update_content_rec, file_id, evaluation_tests_dir, update_content, logger):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tmp_compilation_file = "tmp_compilation.py" + (file_id or "") + str(timestamp)
    if evaluation_tests_dir:
        if rule in applicable_rules:
            logger.info("Checking transformation '%s' with tests.", rule)
            utils.write_file(tmp_compilation_file, update_content_rec)
            recursion_result = complexity.run_pytest(file_id, tmp_compilation_file, evaluation_tests_dir)
            if os.path.exists(tmp_compilation_file):
                os.remove(tmp_compilation_file)
            if "tests_pass" in str(recursion_result):
                return update_content_rec, applicable_rules
            else:
                logger.info("Drop '%s' due to test failure: %s", rule, recursion_result)
                applicable_rules.remove(rule)
                return update_content, applicable_rules
        return update_content, applicable_rules
    else:
        logger.info("No evaluation_tests_dir provided; skipping tests for '%s'.", rule)
        return update_content_rec, applicable_rules


def check_tests_results(rule, applicable_rules, update_content_rec, file_id, evaluation_tests_dir, update_content, logger):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    tmp_compilation_file = "tmp_compilation.py" + (file_id or "") + str(timestamp)
    if not evaluation_tests_dir:
        raise ValueError("No tests provided.")
    if rule in applicable_rules:
        logger.info("Running tests for '%s'...", rule)
        utils.write_file(tmp_compilation_file, update_content_rec)
        recursion_result = complexity.run_pytest(file_id, tmp_compilation_file, evaluation_tests_dir)
        if os.path.exists(tmp_compilation_file):
            os.remove(tmp_compilation_file)
        passed = "tests_pass" in str(recursion_result)
        if not passed:
            logger.info("Tests failed for '%s': %s", rule, recursion_result)
        return passed
    return False


# ---------- apply all rules ----------
def transformation_all_applicable_rules(source_file, applicable_rules, target_file, file_id=None, evaluation_tests_dir=None, logger: Optional[logging.Logger]=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    code = utils.read_file(source_file)
    try:
        update_content, applicable_rules = OP_SIMPLE["change_var_names"](code, applicable_rules)

        for name in ["add_nested_for_out", "add_nested_while_out", "add_nested_if"]:
            cand, applicable_rules = OP_SIMPLE[name](update_content, applicable_rules)
            update_content, applicable_rules = check_tests(name, applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        update_content, applicable_rules = OP_SIMPLE["add_try_except_inside_functions"](update_content, applicable_rules)
        update_content, applicable_rules = OP_SIMPLE["add_else_to_for"](update_content, applicable_rules)
        update_content, applicable_rules = OP_SIMPLE["add_else_to_while"](update_content, applicable_rules)

        cand, applicable_rules = OP_SIMPLE["add_nested_list"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_nested_list", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["transform_range_to_recursion"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("transform_range_to_recursion", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_thread"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_thread", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["replace_with_numpy"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("replace_with_numpy", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_datetime"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_datetime", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_time"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_time", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        update_content, applicable_rules = OP_SIMPLE["add_crypto"](update_content, applicable_rules)
        update_content, applicable_rules = OP_SIMPLE["add_sklearn"](update_content, applicable_rules)

        cand, applicable_rules = OP_SIMPLE["create_functions"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("create_functions", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_decorator"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_decorator", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        if file_id and ("atcoder" in file_id or "codeforces" in file_id):
            cand, applicable_rules = OP_SIMPLE["change_function_names"](update_content, applicable_rules)
            update_content, applicable_rules = check_tests("change_function_names", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_http"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_http", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_scipy"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_scipy", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_base64"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_base64", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        cand, applicable_rules = OP_SIMPLE["add_dateutil"](update_content, applicable_rules)
        update_content, applicable_rules = check_tests("add_dateutil", applicable_rules, cand, file_id, evaluation_tests_dir, update_content, logger)

        update_content, applicable_rules = OP_SIMPLE["changing_AugAssign"](update_content, applicable_rules)

    except Exception as e:
        logger.exception("Error in transformation_all_applicable_rules: %s", e)
    return update_content, applicable_rules


def initialize_whole_benchmark(files_to_transform, genetic_algorithm, evaluation_tests_dir, time_budget, logger: Optional[logging.Logger]=None, apr=False):
    if logger is None:
        logger = logging.getLogger(__name__)
    if genetic_algorithm and files_to_transform:
        ga_whole_benchmark(files_to_transform, evaluation_tests_dir, time_budget, logger, apr)


# ---------- Diff & bug injection ----------
class DocstringRemover(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), (ast.Str, ast.Constant)):
            del node.body[0]
        self.generic_visit(node)
        return node


def remove_docstrings(source_code):
    tree = ast.parse(source_code)
    new_tree = DocstringRemover().visit(tree)
    return ast.unparse(new_tree)


def compare_files_difflib(file1_path, file2_path):
    file1_text = ast.unparse(ast.parse(remove_docstrings(utils.read_file(file1_path)))).split("\n")
    file2_text = ast.unparse(ast.parse(remove_docstrings(utils.read_file(file2_path)))).split("\n")
    diff = list(difflib.Differ().compare(file1_text, file2_text))
    diff_dict = {}
    old_line = None
    for line in diff:
        if line.startswith('-'):
            old_line = line[2:].strip()
        elif line.startswith('+') and old_line:
            diff_dict[old_line] = line[2:].strip()
            old_line = None
    return diff_dict


def get_diff(original_file, new_file):
    return compare_files_difflib(original_file, new_file)


def inject_bugs(files_to_transform, evaluation_tests_dir, json_file, target_dir):
    tmp_dir = target_dir
    results = {}
    for file in files_to_transform:
        name_seed = utils.generate_sha(file)
        target_file = f"{tmp_dir}/{file.split('/')[-1].split('-')[0]}-{name_seed}"
        applicale_apr_rules = get_all_apr_rules(file)
        target_file = apply_multiple_rules(file, target_file, applicale_apr_rules)
        diff_dict = get_diff(file, target_file)
        results[file] = {
            "target_file": target_file,
            "diff": diff_dict,
            "applicale_apr_rules": applicale_apr_rules,
        }
    utils.write_json(json_file, results)
    return json_file


# ---------- misc ----------
def generate_random_string(length=10):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def delete_file_if_exists(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        # (cleaner logs elsewhere)
    except Exception as e:
        pass


def setup_file_info(file_id, file_path, time_budget):
    return {
        "file_id": file_id,
        "original_file": file_path,
        "time_per_iteration": {},
        "used_time": 0,
        "stop_rules": [],
        "unchanged_generations": 0,
        "time_limit": time_budget,
        "previous_pareto_set": set(),
        "potential_individuals": {},
        "rule_sequence_pool": [],
        "current_pareto_set_relative": {},
        "uniq_rules_per_iteration": {},
        "uniq_rules_all_iterations": [],
        "applied_rules": [],
        "result_initialization": False,
        "final_transformations": [],
        "iterations": 0,
        "all_files_per_iteration": {},
        "selected_files_per_iteration": {},
        "termination_reason": None,
    }


def get_time_used(time_dict):
    total = 0
    for iteration in time_dict:
        try:
            total += time_dict[iteration]["end"] - time_dict[iteration]["start"]
        except Exception:
            pass
    return total


def find_most_diverse_rules_item(final_set):
    most_diverse = None
    max_complexity_item = None
    max_unique_rules = 0
    max_complexity = 0
    for item in final_set:
        rules = item['applied_rules']
        complexity = item["relevant_distance_complexity"]
        unique_cnt = len(set(rules))
        if unique_cnt > max_unique_rules:
            max_unique_rules = unique_cnt
            most_diverse = item
        if complexity > max_complexity:
            max_complexity = complexity
            max_complexity_item = item
    return random.choice([most_diverse, max_complexity_item])


# ---------- GA  ----------
def ga_whole_benchmark(files_to_transform, evaluation_tests_dir, time_budget, logger: logging.Logger, apr=False) -> str:
    """
    Run GA over the given files (normally 1).
    Returns the path to the GA run directory under the instance's result dir.
    """
    instance_dir = _instance_dir_from_logger(logger)
    ga_dir = os.path.join(instance_dir, "ga")  # for this instance
    os.makedirs(ga_dir, exist_ok=True)
    os.makedirs(os.path.join(ga_dir, "jsons"), exist_ok=True)
    os.makedirs(os.path.join(ga_dir, ".tmp"), exist_ok=True)
    os.makedirs(os.path.join(ga_dir, "generations"), exist_ok=True)
    os.makedirs(os.path.join(ga_dir, "selection"), exist_ok=True)
    os.makedirs(os.path.join(ga_dir, "no_inter_dependency"), exist_ok=True)

    logger.info("GA artifacts will be saved under: %s", ga_dir)

    programs_to_transform = files_to_transform[:]
    program_id_to_transform = [file.split(".py")[0].split("/")[-1] for file in files_to_transform]
    programs_evolution = {}
    iteration_individuals = {0: []}
    all_individuals = {0: {}}
    individuals = {}
    all_files_applied_rules = {}
    iterations = 0
    result_initialization = False

    budget = time_budget

    # ---- Initial population per file ----
    for file_path in programs_to_transform:
        # try:
        if True:
            file_id = file_path.split(".py")[0].split("/")[-1]
            if file_id not in programs_evolution:
                programs_evolution[file_id] = setup_file_info(file_id, file_path, time_budget)

            if file_path not in individuals:
                initial_rules = get_applicable_rules(file_path, os.path.join(ga_dir, ".tmp"))
                initial_complexity = utils.get_complexity(file_path, "initial")  # no CSV
                initial_readability = utils.get_readability(file_path)
                rd_read = utils.relevant_distance_readability(target_readability, initial_readability)
                rd_comp = utils.relevant_distance_complexity(target_complexity, initial_complexity)
                initial_node = generate_new_node(file_path, initial_rules, [], initial_complexity, initial_readability, rd_read, rd_comp)
                individuals[file_path] = initial_node
                logger.info("Initial file: %s | rd_read: %.4f | rd_comp: %.4f | rules: %s",
                            file_path, rd_read, rd_comp, initial_rules)
                if file_path not in iteration_individuals[0]:
                    iteration_individuals[0].append(file_path)
        # except Exception as e:
        #     logger.exception("Initialization error for %s: %s", file_path, e)

    start_time = time.time()
    logger.info("GA START at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)))
    budget_limit = False

    # ---- Main GA loop ----
    while len(programs_to_transform) > 0:
        if budget_limit:
            break
        if len(iteration_individuals[iterations]) == 0:
            logger.warning("iteration_individuals-%d is empty; stopping.", iterations)
            break

        iterations += 1
        iteration_individuals.setdefault(iterations, [])
        all_individuals.setdefault(iterations, {})
        logger.info("=== Iteration %d ===", iterations)
        logger.info("Files to work on: %d", len(iteration_individuals[iterations - 1]))
        logger.info("Worklist: %s", iteration_individuals[iterations - 1])

        for variant_file in iteration_individuals[iterations - 1]:
            try:
                if variant_file == "target":
                    continue
                if variant_file not in all_files_applied_rules:
                    all_files_applied_rules[variant_file] = []
                iteration_rules = []
                new_individuals = {}
                program_id = variant_file.split(".py")[0].split("/")[-1]
                if program_id not in program_id_to_transform:
                    continue

                # per-file generations dir
                file_gen_dir = os.path.join(ga_dir, "generations", os.path.basename(variant_file).split(".py")[0])
                os.makedirs(file_gen_dir, exist_ok=True)

                applicable_rules = get_applicable_rules(variant_file, os.path.join(ga_dir, ".tmp"))
                if iterations not in programs_evolution[program_id]["time_per_iteration"]:
                    programs_evolution[program_id]["time_per_iteration"][iterations] = {"start": time.time(), "end": None}

                for rule in applicable_rules:
                    logger.info("[Iter %d] Apply rule: %s on %s", iterations, rule, variant_file)
                    if rule in programs_evolution[program_id]["stop_rules"]:
                        logger.info("Rule '%s' in stop_rules, skipping.", rule)
                        continue

                    name_seed = utils.generate_sha(f"{variant_file}_{rule}")
                    new_file = f"{file_gen_dir}/{os.path.basename(variant_file).split('-')[0]}-{name_seed}"
                    if new_file not in all_files_applied_rules:
                        all_files_applied_rules[new_file] = []
                    try:
                        update_content, _ = transformation_single_rule(variant_file, rule, [], new_file)
                        utils.write_file(new_file, update_content)
                        logger.info("Applied rule '%s' -> %s", rule, new_file)
                    except Exception as e:
                        logger.info("Rule '%s' failed: %s", rule, e)
                        continue

                    all_files_applied_rules[new_file].extend(all_files_applied_rules[variant_file])
                    all_files_applied_rules[new_file].append(rule)

                    applied_rules = all_files_applied_rules[new_file]
                    applicable_rules = get_applicable_rules(new_file, os.path.join(ga_dir, ".tmp"))
                    res = check_tests_results(rule, applied_rules, update_content, program_id, evaluation_tests_dir, update_content, logger)
                    if not res:
                        logger.info("Tests failed for rule '%s' on %s", rule, new_file)
                        continue

                    try:
                        new_complexity = utils.get_complexity(new_file, "tmp")  # no CSV
                        new_readability = utils.get_readability(new_file)
                        rd_read = utils.relevant_distance_readability(target_readability, new_readability)
                        rd_comp = utils.relevant_distance_complexity(target_complexity, new_complexity)
                    except Exception as e:
                        logger.info("Metric processing error: %s", e)
                        continue

                    to_exclude_rules, lower_feature = utils.readability_stopping(new_readability, target_readability)
                    programs_evolution[program_id]["stop_rules"].extend(to_exclude_rules)
                    if to_exclude_rules and rule in to_exclude_rules:
                        logger.info("Drop %s due to lower readability (%s) when applying '%s'.", new_file, lower_feature, rule)
                        new_file = variant_file  # revert
                        
                    new_file_backup = new_file
                    # Optionally apply non-complexity-changing rule
                    if apr:
                        random_rule, random_seed = get_applicable_apr(new_file, None)
                    else:
                        random_rule, random_seed = get_applicable_rules_complexity_non_changing(new_file, None)

                    random_rule_applied = False
                    if random_rule is not None:
                        try:
                            name_seed2 = utils.generate_sha(f"{new_file}_{random_rule}")
                            new_file2 = f"{file_gen_dir}/{os.path.basename(new_file).split('-')[0]}-{name_seed2}"
                            if new_file2 not in all_files_applied_rules:
                                all_files_applied_rules[new_file2] = []
                            update_content, _ = transformation_single_rule(new_file, random_rule, [], new_file2)
                            utils.write_file(new_file2, update_content)
                            all_files_applied_rules[new_file2].extend(all_files_applied_rules[new_file])
                            all_files_applied_rules[new_file2].append(random_rule)
                            random_rule_applied = True
                            logger.info("Applied random rule: %s (seed=%s)", random_rule, random_seed)
                        except Exception:
                            pass

                    if random_rule_applied:
                        new_file = new_file2
                        applied_rules = all_files_applied_rules[new_file]
                        applicable_rules = get_applicable_rules(new_file, os.path.join(ga_dir, ".tmp"))
                        if sorted(applied_rules) in [sorted(rs) for rs in programs_evolution[program_id]["rule_sequence_pool"]]:
                            logger.info("Redundant rules sequence: %s", applied_rules)
                            continue
                        res = check_tests_results(random_rule, applied_rules, update_content, program_id, evaluation_tests_dir, update_content, logger)
                        if not res:
                            # revert to prior new_file (before random rule)
                            new_file = new_file_backup
                            logger.info("Random rule '%s' failed tests; revert.", random_rule)
                            applied_rules.remove(random_rule)
                            # continue
                        try:
                            new_complexity = utils.get_complexity(new_file, "tmp")
                            new_readability = utils.get_readability(new_file)
                            rd_read = utils.relevant_distance_readability(target_readability, new_readability)
                            rd_comp = utils.relevant_distance_complexity(target_complexity, new_complexity)
                        except Exception as e:
                            logger.info("Metric processing error after random rule: %s", e)
                            continue
                        to_exclude_rules, lower_feature = utils.readability_stopping(new_readability, target_readability)
                        programs_evolution[program_id]["stop_rules"].extend(to_exclude_rules)
                        if to_exclude_rules and random_rule in to_exclude_rules:
                            logger.info("Random rule '%s' reduced readability; drop.", random_rule)
                            continue

                    logger.info("Candidate: %s | rd_read=%.4f | rd_comp=%.4f | rules=%s",
                                new_file, rd_read, rd_comp, applied_rules)

                    new_variant = generate_new_node(new_file, applicable_rules, applied_rules, new_complexity, new_readability, rd_read, rd_comp)
                    for r in applied_rules:
                        if r not in iteration_rules:
                            iteration_rules.append(r)
                        if r not in programs_evolution[program_id]["uniq_rules_all_iterations"]:
                            programs_evolution[program_id]["uniq_rules_all_iterations"].append(r)

                    new_individuals[new_file] = new_variant
                    individuals[new_file] = new_variant
                    programs_evolution[program_id]["rule_sequence_pool"].append(applied_rules)
                    programs_evolution[program_id].setdefault(iterations, {})
                    all_individuals[iterations][new_file] = utils.read_file(new_file)
                    programs_evolution[program_id][iterations][new_file] = new_variant
                    programs_evolution[program_id]["iterations"] = iterations

            except Exception as e:
                logger.exception("Error while iterating %s: %s", variant_file, e)
                continue

            programs_evolution[program_id]["time_per_iteration"][iterations]["end"] = time.time()
            programs_evolution[program_id]["used_time"] = get_time_used(programs_evolution[program_id]["time_per_iteration"])
            if iterations not in programs_evolution[program_id]["uniq_rules_per_iteration"] and iteration_rules:
                programs_evolution[program_id]["uniq_rules_per_iteration"][iterations] = iteration_rules
            programs_evolution[program_id]["potential_individuals"].update(new_individuals)
            programs_evolution[program_id].setdefault("selected_files_per_iteration", {}).setdefault(iterations, [])

            # Selection for next iteration
            logger.info(f"Selecting individuals for next iteration {iterations} from {len(new_individuals)} candidates.")
            for v in new_individuals:
                logger.info(" - %s, complexity: %s, readability: %s", new_individuals[v]["variant_file"], new_individuals[v]["relevant_distance_complexity"], new_individuals[v]["relevant_distance_readability"])
            variant_iteration_optimal = get_final_pareto_optimal_set(new_individuals)
            logger.info("variant_iteration_optimal_set size: %d", len(variant_iteration_optimal))
            for v in variant_iteration_optimal:
                logger.info(" - %s, complexity: %s, readability: %s", v["variant_file"], v["relevant_distance_complexity"], v["relevant_distance_readability"])
                item = v["variant_file"]
                iteration_individuals[iterations].append(item)
                programs_evolution[program_id]["selected_files_per_iteration"][iterations].append(item)

            if iterations not in programs_evolution[program_id]:
                if programs_evolution[program_id]["original_file"] in programs_to_transform:
                    programs_to_transform.remove(programs_evolution[program_id]["original_file"])
                if program_id in program_id_to_transform:
                    program_id_to_transform.remove(program_id)
                programs_evolution[program_id]["termination_reason"] = f"no new transformations at Iteration {iterations}"
                result_initialization = summary(program_id, programs_evolution, individuals, result_initialization, ga_dir, all_files_applied_rules, evaluation_tests_dir, logger)
            else:
                programs_evolution[program_id].setdefault("all_files_per_iteration", {})[iterations] = programs_evolution[program_id][iterations]

        # Global fronts per program
        for pid in programs_evolution:
            rel_fronts = get_pareto_optimal_set(programs_evolution[pid]["potential_individuals"])
            programs_evolution[pid]["current_pareto_set_relative"] = {v["variant_file"] for v in rel_fronts}

            if programs_evolution[pid]["current_pareto_set_relative"] == programs_evolution[pid]["previous_pareto_set"]:
                programs_evolution[pid]["unchanged_generations"] += 1
            else:
                programs_evolution[pid]["unchanged_generations"] = 0
                programs_evolution[pid]["previous_pareto_set"] = programs_evolution[pid]["current_pareto_set_relative"]

            if programs_evolution[pid]["unchanged_generations"] == 3:
                if programs_evolution[pid]["original_file"] in programs_to_transform:
                    programs_to_transform.remove(programs_evolution[pid]["original_file"])
                if pid in program_id_to_transform:
                    program_id_to_transform.remove(pid)
                programs_evolution[pid]["termination_reason"] = "unchanged_generations"
                result_initialization = summary(pid, programs_evolution, individuals, result_initialization, ga_dir, all_files_applied_rules, evaluation_tests_dir, logger)
                continue

            if programs_evolution[pid]["used_time"] >= programs_evolution[pid]["time_limit"]:
                if programs_evolution[pid]["original_file"] in programs_to_transform:
                    programs_to_transform.remove(programs_evolution[pid]["original_file"])
                if pid in program_id_to_transform:
                    program_id_to_transform.remove(pid)
                programs_evolution[pid]["termination_reason"] = "time limit"
                result_initialization = summary(pid, programs_evolution, individuals, result_initialization, ga_dir, all_files_applied_rules, evaluation_tests_dir, logger)
                continue

        # JSON only
        utils.write_json(os.path.join(ga_dir, "jsons", f"iteration_individuals{iterations}.json"), iteration_individuals[iterations])
        utils.write_json(os.path.join(ga_dir, "jsons", f"all_individuals{iterations}.json"), all_individuals[iterations])

        if time.time() - start_time >= budget:
            logger.info("Terminating GA due to time limit.")
            utils.write_json(os.path.join(ga_dir, "jsons", f"iteration_individuals{iterations}.json"), iteration_individuals[iterations])
            utils.write_json(os.path.join(ga_dir, "jsons", f"all_individuals{iterations}.json"), all_individuals[iterations])
            budget_limit = True
            break

    utils.write_json(os.path.join(ga_dir, "jsons", "all_files_applied_rules.json"), all_files_applied_rules)
    for pid in program_id_to_transform:
        programs_evolution[pid]["termination_reason"] = "time limit"
        result_initialization = summary(pid, programs_evolution, individuals, result_initialization, ga_dir, all_files_applied_rules, evaluation_tests_dir, logger)

    utils.dump_json(os.path.join(ga_dir, "jsons", "programs_evolution.json"), programs_evolution)
    logger.info("GA END at %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    return ga_dir  # path to GA directory under instance result dir


# ---------- GA summary & best pick ----------
def get_newclass(code):
    newclass = []
    if "from newClass" in code:
        for line in code.split("\n"):
            if "from newClass" in line:
                newfile = line.split("from")[-1].split("import")[0].strip() + ".py"
                if newfile not in newclass:
                    newclass.append(newfile)
    return newclass


def summary(program_id, programs_evolution, individuals, result_initialization, ga_dir, all_files_applied_rules, evaluation_tests_dir, logger: logging.Logger):
    current_pareto_set_relative = programs_evolution[program_id]["current_pareto_set_relative"]
    potential_individuals = programs_evolution[program_id]["potential_individuals"]
    source_file = programs_evolution[program_id]["original_file"]
    uniq_rules_per_iteration = programs_evolution[program_id]["uniq_rules_per_iteration"]
    uniq_rules_all_iterations = programs_evolution[program_id]["uniq_rules_all_iterations"]
    termination_reason = programs_evolution[program_id]["termination_reason"]

    final_set = get_final_pareto_optimal_set(potential_individuals)
    most_diverse_item = find_most_diverse_rules_item(final_set)

    original_content = ""
    newclass = []
    if most_diverse_item is not None:
        original_content = utils.read_file(most_diverse_item["variant_file"])
        original_rules = copy.deepcopy(most_diverse_item["applied_rules"])
        update_content, most_diverse_item["applied_rules"] = transformation_single_rule(
            most_diverse_item["variant_file"], "move_functions_into_new_class",
            most_diverse_item["applied_rules"], most_diverse_item["variant_file"]
        )
        if "move_functions_into_new_class" in most_diverse_item["applied_rules"]:
            res = check_tests_results("move_functions_into_new_class", most_diverse_item["applied_rules"], update_content,
                                      program_id, evaluation_tests_dir, update_content, logger)
            if res:
                utils.write_file(most_diverse_item["variant_file"], update_content)
                newclass = get_newclass(update_content)
            else:
                most_diverse_item["applied_rules"] = original_rules
                utils.write_file(most_diverse_item["variant_file"], original_content)

    result = get_statistics(final_set)

    # JSON summary
    res = {
        "source_file": source_file,
        "termination_reason": termination_reason,
        "initial_relevant_distance_complexity": individuals[source_file]["relevant_distance_complexity"],
        "initial_relevant_distance_readability": individuals[source_file]["relevant_distance_readability"],
        "#initial applicable operators": len(individuals[source_file]["applicable_rules"]),
        "#all applicable operators": len(uniq_rules_all_iterations),
        "#uniq_rules_per_iteration": [(it, len(uniq_rules_per_iteration[it])) for it in uniq_rules_per_iteration],
        "operators used per iteration": uniq_rules_per_iteration,
        "iterations": programs_evolution[program_id]["iterations"],
        "time": programs_evolution[program_id]["used_time"],
        "#final_population_size": len(potential_individuals),
        "#pareto_optimal_set with complexity higher than median": len(current_pareto_set_relative),
        "#final_transformations": len(final_set),
        "final_transformations": final_set,
        "final_selection": most_diverse_item["variant_file"] if most_diverse_item is not None else None,
        "final_selection_original_content": original_content,
        "final_selection_class_dependencies": newclass,
        "avg_complexity": result['avg_complexity'],
        "avg_readability": result['avg_readability'],
        "min_complexity": result['min_complexity'],
        "min_readability": result['min_readability'],
        "max_complexity": result['max_complexity'],
        "max_readability": result['max_readability'],
        "final_selection_rules": most_diverse_item["applied_rules"] if most_diverse_item is not None else None,
        "final_selection_complexity": most_diverse_item["relevant_distance_complexity"] if most_diverse_item is not None else None,
        "final_selection_readability": most_diverse_item["relevant_distance_readability"] if most_diverse_item is not None else None,
        "all_files_per_iteration": programs_evolution[program_id]["all_files_per_iteration"],
        "selected_files_per_iteration": programs_evolution[program_id]["selected_files_per_iteration"],
    }

    # JSON under GA dir
    src_base = os.path.basename(source_file).split(".py")[0]
    json_summary_path = os.path.join(ga_dir, "jsons", f"{src_base}.json")
    utils.write_json(json_summary_path, res)
    logger.info("Wrote GA summary: %s", json_summary_path)

    # Save chosen selection + backup original content + copy class dependencies
    if most_diverse_item is not None and most_diverse_item["variant_file"]:
        shutil.copy(most_diverse_item["variant_file"], os.path.join(ga_dir, "selection"))
        backup_file = os.path.join(ga_dir, "no_inter_dependency", os.path.basename(most_diverse_item["variant_file"]))
        utils.write_file(backup_file, res["final_selection_original_content"])
        for file in newclass:
            class_path = os.path.join(os.path.dirname(most_diverse_item["variant_file"]), file)
            if os.path.exists(class_path):
                shutil.copy(class_path, os.path.join(ga_dir, "selection"))
    return True


def _run_ga_and_pick_best(files: List[str], evaluation_tests_dir: Optional[str], time_budget: float, apr: bool, logger: logging.Logger) -> Optional[Tuple[str, List[str]]]:
    """
    Runs GA and returns (best_variant_file_path, applied_rules_list) for the single input file.
    If nothing selected, returns None.
    """
    if not files:
        return None
    ga_dir = ga_whole_benchmark(files, evaluation_tests_dir, time_budget, logger, apr)
    src_base = os.path.basename(files[0]).split(".py")[0]
    summary_path = os.path.join(ga_dir, "jsons", f"{src_base}.json")
    if not os.path.exists(summary_path):
        logger.warning("GA did not produce summary at %s", summary_path)
        return None
    data = utils.read_json(summary_path)
    best_path = data.get("final_selection")
    applied_rules = data.get("final_selection_rules")
    if best_path and os.path.exists(best_path):
        logger.info("Best GA selection: %s", best_path)
        return best_path, applied_rules or []
    logger.warning("GA summary present but best file missing; falling back to original.")
    return files[0], []
