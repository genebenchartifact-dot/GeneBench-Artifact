import ast
import random

def insert_datetime_import(node):
    has_datetime = any(isinstance(import_node, ast.Import) and
                       any(alias.name == 'datetime' for alias in import_node.names)
                       for import_node in node.body if isinstance(import_node, ast.Import))
    if not has_datetime:
        datetime_import = ast.Import(names=[ast.alias(name='datetime', asname=None)])
        node.body.insert(0, datetime_import)
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

def insert_datetime_now_call(node):
    """Inserts a datetime.datetime.now() call at the start of the main code block."""
    now_call = ast.Expr(value=ast.Call(
        func=ast.Attribute(
            value=ast.Attribute(
                value=ast.Name(id='datetime', ctx=ast.Load()),
                attr='datetime',
                ctx=ast.Load()
            ),
            attr='now',
            ctx=ast.Load()
        ),
        args=[],
        keywords=[]
    ))

    insert_call(node, now_call)
 
    # insert_index = random.randint(1, len(node.body))
    # node.body.insert(insert_index, now_call)
    return node


def add_datetime(python_code, applicable_rules):
    parsed_code = ast.parse(python_code)
    if_applicable = insert_datetime_import(parsed_code)
    if if_applicable:
        update_node = insert_datetime_now_call(parsed_code)
        update_content = ast.unparse(update_node)
        applicable_rules.append("add_datetime")
    return update_content, applicable_rules
    

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_datetime(python_code, applicable_rules)
    return update_content, applicable_rules