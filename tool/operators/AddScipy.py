import ast
import random

def insert_scipy_import(node):
    has_scipy = any(isinstance(import_node, (ast.Import, ast.ImportFrom)) and
                    ('ttest_ind' in [alias.name for alias in import_node.names]
                     if isinstance(import_node, (ast.Import, ast.ImportFrom))
                     else import_node.module == 'scipy.stats')
                    for import_node in node.body)
    if not has_scipy:
        scipy_import = ast.ImportFrom(module='scipy.stats', names=[ast.alias(name='ttest_ind', asname=None)], level=0)
        node.body.insert(0, scipy_import)
        return True
    return False

def insert_call(node, call):
    functions = []

    for n in node.body:
        if isinstance(n, ast.FunctionDef):
            functions.append(n)
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    functions.append(m)

    if functions:
        target_function = random.choice(functions)
        if len(target_function.body) >= 1:
            insert_index = random.randint(0, len(target_function.body) - 1)
        else:
            insert_index = 0
        target_function.body.insert(insert_index, call)
    else:
        insert_index = random.randint(0, len(node.body) - 1)
        node.body.insert(insert_index, call)

def insert_ttest_call(node):
    """scipy.stats.ttest_ind(list1, list2)"""
    ttest_call = ast.Expr(value=ast.Call(
        func=ast.Name(id='ttest_ind', ctx=ast.Load()),
        args=[
            ast.List(elts=[ast.Num(n=random.randint(1, 100)), ast.Num(n=random.randint(1, 100)), ast.Num(n=random.randint(1, 100))], ctx=ast.Load()),
            ast.List(elts=[ast.Num(n=random.randint(1, 100)), ast.Num(n=random.randint(1, 100)), ast.Num(n=random.randint(1, 100))], ctx=ast.Load())
        ],
        keywords=[]
    ))

    insert_call(node, ttest_call)
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, ttest_call)
    return node

def add_scipy(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_scipy_import(parsed_code)
    if if_applicable:
        updated_node = insert_ttest_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_scipy")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_scipy(python_code, applicable_rules)
    return updated_content, applicable_rules