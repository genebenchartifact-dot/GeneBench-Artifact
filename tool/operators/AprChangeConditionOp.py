import ast
import random

class ChangeConditionOps(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
    
    def visit_BoolOp(self, node):
        if self.counter > 5:
            return node
        if isinstance(node.op, ast.And):
            node.op = ast.Or()
            self.counter += 1
        elif isinstance(node.op, ast.Or):
            node.op = ast.And()
            self.counter += 1
        return self.generic_visit(node)
    
    def visit_UnaryOp(self, node):
        if self.counter > 5:
            return node
        if isinstance(node.op, ast.Not):
            self.counter += 1
            return node.operand
        return self.generic_visit(node)
    
    def visit_If(self, node):
        if self.counter < 5 or ast.unparse(node) == "if __name__ == '__main__':":
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.counter += 1
            return node
        return self.generic_visit(node)
    
    def visit_While(self, node):
        if self.counter < 5: 
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
            self.counter += 1
            return node
        return self.generic_visit(node)

def apr_change_condition_op(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = ChangeConditionOps()
    new_root = transformer.visit(root)
    if transformer.counter > 0:
        applicable_rules.append("apr_change_condition_op")
    update_content = ast.unparse(new_root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = apr_change_condition_op(python_code, applicable_rules)
    return update_content, applicable_rules