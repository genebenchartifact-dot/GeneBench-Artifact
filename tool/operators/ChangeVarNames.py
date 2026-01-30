import ast

class RenameVariable(ast.NodeTransformer):
    def __init__(self, variable_map):
        self.variable_map = variable_map
        self.counter = 0

    def visit_Name(self, node):
        if node.id in self.variable_map:
            node.id = self.variable_map[node.id]
            self.counter += 1
        return self.generic_visit(node)

    def visit_arg(self, node):
        if node.arg in self.variable_map:
            node.arg = self.variable_map[node.arg]
            self.counter += 1
        return self.generic_visit(node)

def generate_new_name(old_name, existing_names):
    index = 1
    new_name = f"new{old_name}_{index}"
    while new_name in existing_names:
        index += 1
        new_name = f"new{old_name}_{index}"
    return new_name

def change_var_names(python_code, applicable_rules):
    tree = ast.parse(python_code)
    all_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and "new" not in node.id}
    
    variable_map = {}
    for name in all_names:
        if name not in variable_map:
            new_name = generate_new_name(name, all_names | set(variable_map.values()))
            variable_map[name] = new_name
            break
    # print(variable_map)
    transformer = RenameVariable(variable_map)
    new_tree = transformer.visit(tree)
    if transformer.counter > 0:
        applicable_rules.append("change_var_names")
    update_content = ast.unparse(new_tree)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = change_var_names(python_code, applicable_rules)
    return update_content, applicable_rules