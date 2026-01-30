import ast


decorator_code = """
def my_decorator(func):
    def dec_result(*args, **kwargs):
        res = func(*args, **kwargs)
        return res
    return dec_result
"""

class IfAdded(ast.NodeVisitor):
    def __init__(self):
        self.my_decorator_exists = False
        
    def visit_FunctionDef(self, node):
        if node.name == "my_decorator":
            self.my_decorator_exists = True
        return node


class AddDecorator(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
        self.my_decorator_exists = False

    def visit_FunctionDef(self, node):
        if node.name == "my_decorator":
            self.my_decorator_exists = True
            return node
        if self.counter < 1 and node.name != "my_decorator":
            for decorator in node.decorator_list:
                if decorator.id == "my_decorator":
                    # print("Already applied operator add_decorator for node: {}!".format(node.name))
                    return node
            decorator = ast.Name(id='my_decorator', ctx=ast.Load())
            node.decorator_list.insert(0, decorator)
            self.counter += 1
        return node

def add_decorator(python_code, applicable_rules):
    decorator_ast = ast.parse(decorator_code)
    root = ast.parse(python_code)
    
    check_if_exists = IfAdded()
    checker = check_if_exists.visit(root)
    if check_if_exists.my_decorator_exists == False:
        transformer = AddDecorator()
        modified_ast = transformer.visit(root)
        if transformer.my_decorator_exists == False:
            root = ast.Module(body=decorator_ast.body + modified_ast.body, type_ignores=[])
        if transformer.counter > 0:
            applicable_rules.append("add_decorator")
        update_content = ast.unparse(root)
    else:
        print("Decorator already added!")
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_decorator(python_code, applicable_rules)
    return update_content, applicable_rules