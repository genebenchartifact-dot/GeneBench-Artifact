import ast
import random

def insert_cryptography_import(node):
    has_cryptography = any((isinstance(import_node, ast.Import) or isinstance(import_node, ast.ImportFrom)) and
                           any(alias.name == 'Fernet' for alias in import_node.names)
                           for import_node in node.body if isinstance(import_node, ast.Import) or isinstance(import_node, ast.ImportFrom))
    if not has_cryptography:
        cryptography_import = ast.ImportFrom(module='cryptography.fernet', names=[ast.alias(name='Fernet', asname=None)], level=0)
        node.body.insert(0, cryptography_import)
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

def insert_cryptography_key_gen_call(node):
    key_gen_call = ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='Fernet', ctx=ast.Load()),
            attr='generate_key',
            ctx=ast.Load()
        ),
        args=[],
        keywords=[]
    ))

    insert_call(node, key_gen_call)
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, key_gen_call)
    return node

def add_cryptography(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_cryptography_import(parsed_code)
    if if_applicable:
        updated_node = insert_cryptography_key_gen_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_crypto")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_cryptography(python_code, applicable_rules)
    return updated_content, applicable_rules
