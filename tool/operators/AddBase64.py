import ast
import random

def generate_random_digit_string(length=20):
    digits = '0123456789'
    random_digits = random.choices(digits, k=length)
    return ''.join(random_digits)

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


def insert_base64_import(node):
    has_base64 = any(isinstance(import_node, ast.Import) and
                     any(alias.name == 'base64' for alias in import_node.names)
                     for import_node in node.body if isinstance(import_node, ast.Import))
    if not has_base64:
        base64_import = ast.Import(names=[ast.alias(name='base64', asname=None)])
        node.body.insert(0, base64_import)
        return True
    return False

def insert_base64_encode_call(node):
    """Inserts a base64.b64encode call at the start of the main code block."""
    encode_call = ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Name(id='base64', ctx=ast.Load()),
            attr='b64encode',
            ctx=ast.Load()
        ),
        args=[ast.Bytes(s=generate_random_digit_string().encode('utf-8'))],
        keywords=[]
    ))
    
    insert_call(node, encode_call)
        
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, encode_call)
    return node

def add_base64(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_base64_import(parsed_code)
    if if_applicable:
        updated_node = insert_base64_encode_call(parsed_code)
        updated_content = ast.unparse(updated_node)
        applicable_rules.append("add_base64")
    return updated_content, applicable_rules

def init(python_code, applicable_rules):
    updated_content, applicable_rules = add_base64(python_code, applicable_rules)
    return updated_content, applicable_rules
