import ast
import random

confusing_names = [ "Zero", "One", "Two", "Three", "Four", "Five", "six", \
            "Seven", "Eight", "Nine", "Ten"]

class ReplaceConstantsWithVariables(ast.NodeTransformer):
    def __init__(self, max_num):
        super().__init__()
        self.value_map = {}
        self.max_num = max_num
        self.counter = 1
        self.declarations = []
        self.transformed_constants = []
        
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in confusing_names and isinstance(node.value, ast.Constant):
                self.transformed_constants.append(node.value.value)
        return node

    def visit_Constant(self, node):
        if self.counter > self.max_num:
            return node
        # Generate a unique variable name for each unique constant
        if node.value not in self.value_map and node.value not in self.transformed_constants:
            if node.value in range(0,11):
                var_name = confusing_names[int((node.value + 1)%10)]
            else:
                idx = random.randint(0, 10)
                var_name = confusing_names[(idx)%10] + "_" + str(node.lineno + 1)

            self.value_map[node.value] = var_name
            
            # Create a new assignment for the variable initialization
            value_node = ast.Constant(value=node.value)
            target_node = ast.Name(id=var_name, ctx=ast.Store())
            assign_node = ast.Assign(targets=[target_node], value=value_node, lineno=node.lineno)
            self.declarations.append(assign_node)
            self.counter += 1
        
            # Replace the constant node with a variable node
            return ast.Name(id=self.value_map[node.value], ctx=ast.Load())
        return node

def introduce_confusing_vars(python_code, applicable_rules, max_num):
    root = ast.parse(python_code)
    transformer = ReplaceConstantsWithVariables(max_num)
    new_tree = transformer.visit(root)
    new_tree.body = transformer.declarations + new_tree.body
    if transformer.counter > 1:
        applicable_rules.append("introduce_confusing_vars")
    update_content = ast.unparse(new_tree)
    return update_content, applicable_rules

def init(python_code, applicable_rules, max_num):
    update_content, applicable_rules = introduce_confusing_vars(python_code, applicable_rules, max_num)
    return update_content, applicable_rules