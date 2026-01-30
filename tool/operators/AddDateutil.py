import ast
import random
from datetime import datetime

def get_current_timestamp():
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

def insert_dateutil_import(node):
    has_dateutil = any(isinstance(import_node, (ast.Import, ast.ImportFrom)) and
                       ('parse' in [alias.name for alias in import_node.names]
                        if isinstance(import_node, (ast.Import, ast.ImportFrom))
                        else import_node.module == 'dateutil.parser')
                       for import_node in node.body)
    if not has_dateutil:
        dateutil_import = ast.ImportFrom(module='dateutil.parser', names=[ast.alias(name='parse', asname=None)], level=0)
        node.body.insert(0, dateutil_import)
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

def insert_dateutil_parse_call(node):
    parse_call = ast.Expr(value=ast.Call(
        func=ast.Name(id='parse', ctx=ast.Load()),
        args=[ast.Str(s=get_current_timestamp())],
        keywords=[]
    ))

    insert_call(node, parse_call)
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, parse_call)
    return node

def add_dateutil(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_dateutil_import(parsed_code)
    if if_applicable:
        updated_node = insert_dateutil_parse_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_dateutil")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_dateutil(python_code, applicable_rules)
    return updated_content, applicable_rules
