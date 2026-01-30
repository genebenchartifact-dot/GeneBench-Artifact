import ast

"""Transform primitive variables with nested lists. 
"""

class TransformList(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
        
    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.List) or isinstance(node.value, ast.ListComp) or isinstance(node.value, ast.Constant):
            if self.counter < 1:
                node.value = ast.Subscript(
                    value=ast.List(elts=[node.value], ctx=ast.Load()),
                    slice=ast.Index(value=ast.Constant(value=0)),
                    ctx=ast.Load()
                )
                self.counter += 1
        return node

    def visit_AugAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.List) or isinstance(node.value, ast.ListComp) or isinstance(node.value, ast.Constant):
            if self.counter < 1:
                node.value = ast.Subscript(
                    value=ast.List(elts=[node.value], ctx=ast.Load()),
                    slice=ast.Index(value=ast.Constant(value=0)),
                    ctx=ast.Load()
                )
                self.counter += 1
        return node

def add_nested_list(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = TransformList()
    modified_ast = transformer.visit(root)
    if transformer.counter > 0:
        applicable_rules.append("add_nested_list")
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    # print(python_code)
    update_content, applicable_rules = add_nested_list(python_code, applicable_rules)
    return update_content, applicable_rules