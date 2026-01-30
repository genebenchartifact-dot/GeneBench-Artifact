import ast
import random

class ChangeReturn(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
        
    def visit_Return(self, node):
        if isinstance(node.value, ast.Constant) and self.counter < 5:
            node.value.n = -node.value.n+10
            self.counter += 1
            return node
        elif isinstance(node.value, ast.Name) and self.counter < 5:
            node.value = ast.UnaryOp(op=ast.Not(), operand=node.value)
            self.counter += 1
            return node
        else:
            node.value = ast.UnaryOp(op=ast.Not(), operand=node.value)
            # print(ast.unparse(node))
            self.counter += 1
            return node
        return self.generic_visit(node)

def apr_change_return(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = ChangeReturn()
    new_root = transformer.visit(root)
    if transformer.counter > 0:
        applicable_rules.append("apr_change_return")
    update_content = ast.unparse(new_root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = apr_change_return(python_code, applicable_rules)
    return update_content, applicable_rules