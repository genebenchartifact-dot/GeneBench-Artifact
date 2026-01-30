import ast
import random

def insert_time_import(node):
    has_time = any(isinstance(import_node, ast.Import) and
                   any(alias.name == 'time' for alias in import_node.names)
                   for import_node in node.body if isinstance(import_node, ast.Import))
    if not has_time:
        time_import = ast.Import(names=[ast.alias(name='time', asname=None)])
        node.body.insert(0, time_import)
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

def insert_time_sleep_call(node):
    sleep_call = ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='time', ctx=ast.Load()),
            attr='sleep',
            ctx=ast.Load()
        ),
        args=[ast.Constant(value=round(random.uniform(0, 0.3),2))],
        keywords=[]
    ))
    
    insert_call(node, sleep_call)
    
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, sleep_call)
    return node

def add_time(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_time_import(parsed_code)
    if if_applicable:
        updated_node = insert_time_sleep_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_time")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_time(python_code, applicable_rules)
    return updated_content, applicable_rules
