import argparse
import ast
import datetime
import logging
import os
import subprocess

import utils
import complexity
import python_transformer


def parse_args():
    parser = argparse.ArgumentParser(description="Python code transformation framework.")
    parser.add_argument("--source_file", required=True, help="Source Python file to transform.")
    parser.add_argument("--target_file", help="Target file to save transformed output.")
    parser.add_argument("--file_id", required=True, help="Unique ID for logging and output files.")
    parser.add_argument("--evaluation_tests_dir", help="Directory of test files.")
    parser.add_argument("--rule", dest="single_rule", help="Apply a specific transformation rule.")
    parser.add_argument("--genetic_algorithm", action='store_true', help="Use genetic algorithm.")
    parser.add_argument("--inject_bugs", action='store_true', help="Inject bugs into the source file.")
    parser.add_argument("--time_budget", type=float, default=3600, help="Genetic algorithm time budget (seconds).")
    parser.add_argument("--result_dir", required=True, help="Directory to store per-instance artifacts.")
    return parser.parse_args()


def setup_logger(log_path):
    """Create a per-instance logger writing to log_path and also to stdout."""
    logger = logging.getLogger(log_path)  # unique per instance
    logger.setLevel(logging.INFO)

    # Clear existing handlers for idempotence
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def clear_content(file_path, logger):
    code = utils.clear_string(utils.read_file(file_path))
    tree = ast.parse(code)
    formatted = ast.unparse(tree)
    utils.write_file(file_path, formatted)
    logger.info("Normalized formatting: %s", file_path)


def autopep_format(file_path, logger):
    try:
        subprocess.run(["autopep8", "--in-place", file_path], check=True)
        logger.info("autopep8 formatted: %s", file_path)
    except Exception as e:
        logger.warning("autopep8 failed: %s", e)


def precheck_tests(file_id, path, test_dir, logger):
    if not test_dir:
        return None
    logger.info("Running pre-transformation tests…")
    result = complexity.run_pytest(file_id, path, test_dir)
    if "tests_pass" not in str(result):
        logger.warning("Original tests failed. Skipping transformation.")
        return result
    logger.info("Tests passed.")
    return result


def run_transformation(args):
    # Per-instance directory (all artifacts live here)
    instance_dir = os.path.join(args.result_dir, args.file_id)
    os.makedirs(instance_dir, exist_ok=True)

    log_path = os.path.join(instance_dir, f"{args.file_id}.log")
    logger = setup_logger(log_path)

    logger.info("Start processing %s", args.file_id)
    start_time = datetime.datetime.now()

    # Default target goes to this instance dir
    if args.target_file:
        target_file = args.target_file
    else:
        base = os.path.basename(args.source_file).replace(".py", "_transformed.py")
        target_file = os.path.join(instance_dir, base)
    args.target_file = target_file  # keep args consistent

    clear_content(args.source_file, logger)

    before_tests = precheck_tests(args.file_id, args.source_file, args.evaluation_tests_dir, logger)
    if args.evaluation_tests_dir and "tests_pass" not in str(before_tests):
        return record_result(args, start_time, None, [], "Original tests failure",
                             before_tests, None, None, logger, instance_dir)

    if args.inject_bugs:
        logger.info("Injecting bugs…")
        bugs_json = os.path.join(instance_dir, f"{args.file_id}_bugs.json")
        python_transformer.inject_bugs(
            [args.source_file],
            args.evaluation_tests_dir,
            bugs_json,
            instance_dir
        )
        logger.info("Bug injection results saved to %s", bugs_json)
        return

    # ---- MAIN CALL (GA inside python_transformer when requested) ----
    transformed_code, applied_rules = python_transformer.initialize(
        source_file=args.source_file,
        single_rule=args.single_rule,
        genetic_algorithm=args.genetic_algorithm,
        target_file=args.target_file,
        file_id=args.file_id,
        evaluation_tests_dir=args.evaluation_tests_dir,
        logger=logger,
        time_budget=args.time_budget,
    )

    patch_path = None
    after_tests = None

    if transformed_code is not None and applied_rules is not None:
        utils.write_file(args.target_file, transformed_code)
        logger.info("Saved transformed code to %s", args.target_file)
        autopep_format(args.target_file, logger)

        diff_output, diff_path = utils.diff(args.source_file, args.target_file)
        patch_path = os.path.join(instance_dir, f"{args.file_id}.patch")
        utils.write_file(patch_path, diff_output)
        logger.info("Saved diff to %s", patch_path)

        if args.evaluation_tests_dir:
            after_tests = complexity.run_pytest(args.file_id, args.target_file, args.evaluation_tests_dir)
            logger.info("Post-transformation tests complete.")

    return record_result(
        args, start_time, transformed_code, applied_rules, None,
        before_tests, after_tests, patch_path, logger, instance_dir
    )


def record_result(args, start, code, rules, exception, before, after, patch_path, logger, instance_dir):
    elapsed = (datetime.datetime.now() - start).total_seconds()
    result = {
        "file_id": args.file_id,
        "source_file": args.source_file,
        "target_file": args.target_file,
        "evaluation_tests_dir": args.evaluation_tests_dir,
        "single_rule": args.single_rule,
        "genetic_algorithm": args.genetic_algorithm,
        "patch_path": patch_path,
        "applicable_rules": rules,
        "exception": exception,
        "total_time": elapsed,
        "test_results_before": before,
        "test_results_after": after,
        "diff_output": patch_path,
        "original_code": utils.read_file(args.source_file),
        "transformed_code": code,
    }

    out_path = os.path.join(instance_dir, f"{args.file_id}.json")
    utils.write_json(out_path, result)

    logger.info("Finished %s in %.2fs", args.file_id, elapsed)
    logger.info("Saved result JSON to %s", out_path)
    if rules:
        logger.info("Applied rules:")
        for rule in rules:
            logger.info(" - %s", rule)

    return result


if __name__ == "__main__":
    args = parse_args()
    run_transformation(args)
