import ast
import random

def insert_sklearn_import(node):
    has_sklearn = any(isinstance(import_node, (ast.Import, ast.ImportFrom)) and
                      ('shuffle' in [alias.name for alias in import_node.names]
                       if isinstance(import_node, (ast.Import, ast.ImportFrom))
                       else import_node.module == 'sklearn.utils')
                      for import_node in node.body)
    if not has_sklearn:
        sklearn_import = ast.ImportFrom(module='sklearn.utils', names=[ast.alias(name='shuffle', asname=None)], level=0)
        node.body.insert(0, sklearn_import)
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

def insert_sklearn_shuffle_call(node):
    """sklearn.utils.shuffle([1, 2, 3])"""
    shuffle_call = ast.Expr(value=ast.Call(
        func=ast.Name(id='shuffle', ctx=ast.Load()),
        args=[ast.List(elts=[ast.Constant(value=random.randint(1, 100)), ast.Constant(value=random.randint(1, 100)), ast.Constant(value=random.randint(1, 100))], ctx=ast.Load())],
        keywords=[]
    ))

    insert_call(node, shuffle_call)
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, shuffle_call)
    return node

def add_sklearn(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_sklearn_import(parsed_code)
    if if_applicable:
        updated_node = insert_sklearn_shuffle_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_sklearn")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_sklearn(python_code, applicable_rules)
    return updated_content, applicable_rules