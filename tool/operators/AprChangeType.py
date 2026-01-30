import ast
import random

class ChangeType(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
        
    def visit_Num(self, node):
        if isinstance(node.n, int) and self.counter < 5:
            choices = [str(node.n), bool(node.n % 2)]
            node.n = random.choice(choices)
            self.counter += 1
        elif isinstance(node.n, float) and self.counter < 5:
            choices = [str(node.n), bool(node.n % 2)]
            node.n = random.choice(choices)
            self.counter += 1
        return self.generic_visit(node)
    
    def visit_Constant(self, node):
        if self.counter < 5 and "intra-dependencies" not in ast.unparse(node):
            node.value = str(random.randint(1000, 9999))
            self.counter += 1
        if node.value == None:
            node.value = str(random.randint(1000, 9999))        
        return self.generic_visit(node)
    
    # def visit_Name(self, node):
    #     if self.counter < 5 and not ast.unparse(node).endswith("_new"):
    #         node.id += '_new'
    #         self.counter += 1
    #     return self.generic_visit(node)

def apr_change_type(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = ChangeType()
    new_root = transformer.visit(root)
    if transformer.counter > 0:
        applicable_rules.append("apr_change_type")
    update_content = ast.unparse(new_root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = apr_change_type(python_code, applicable_rules)
    return update_content, applicable_rules