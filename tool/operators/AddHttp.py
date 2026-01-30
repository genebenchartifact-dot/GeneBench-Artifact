import ast
import random

def insert_http_import(node):
    has_http = any(isinstance(import_node, (ast.Import, ast.ImportFrom)) and
                   ('HTTPConnection' in [alias.name for alias in import_node.names]
                    if isinstance(import_node, (ast.Import, ast.ImportFrom))
                    else import_node.module == 'http.client')
                   for import_node in node.body)
    if not has_http:
        http_import = ast.ImportFrom(module='http.client', names=[ast.alias(name='HTTPConnection', asname=None)], level=0)
        node.body.insert(0, http_import)
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

def insert_http_call(node):
    """Inserts an HTTPConnection call that creates a connection to example.com."""
    http_call = ast.Expr(value=ast.Call(
        func=ast.Name(id='HTTPConnection', ctx=ast.Load()),
        args=[
            ast.Str(s='google.com')
        ],
        keywords=[
            ast.keyword(arg='port', value=ast.Num(n=80))
        ]
    ))

    insert_call(node, http_call)
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, http_call)
    return node

def add_http(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_http_import(parsed_code)
    if if_applicable:
        updated_node = insert_http_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_http")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_http(python_code, applicable_rules)
    return updated_content, applicable_rules
