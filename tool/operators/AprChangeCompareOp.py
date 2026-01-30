import ast
import random

class ChangeCompareOps(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
    
    def visit_Compare(self, node):
        replacements = {
            ast.Lt: [ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq],
            ast.LtE: [ast.Gt, ast.Lt, ast.GtE, ast.Eq, ast.NotEq],
            ast.Gt: [ast.Lt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq],
            ast.GtE: [ast.Lt, ast.LtE, ast.Gt, ast.Eq, ast.NotEq],
            ast.Eq: [ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
            ast.NotEq: [ast.Eq, ast.Lt, ast.LtE, ast.Gt, ast.GtE],
        }

        if type(node.ops[0]) in replacements and self.counter <= 5:
            node.ops[0] = random.choice(replacements[type(node.ops[0])])()
            self.counter += 1
        
        return self.generic_visit(node)

def apr_change_compare_op(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = ChangeCompareOps()
    new_root = transformer.visit(root)
    if transformer.counter > 0:
        applicable_rules.append("apr_change_compare_op")
    update_content = ast.unparse(new_root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = apr_change_compare_op(python_code, applicable_rules)
    return update_content, applicable_rules