import ast

class PromoteLocalVarsToGlobals(ast.NodeTransformer):
    def __init__(self):
        self.global_vars = {}  # To hold unique global variables
        self.func_local_cars = {}
        self.current_function_name = ""

    def visit_FunctionDef(self, node):
        self.current_function_name = node.name  # Track the current function
        self.generic_visit(node)  # Visit all nodes within the function
        return node

    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and self.current_function_name != "" \
            and not isinstance(node.value, ast.Subscript):
            var_name = node.targets[0].id
            unique_var_name = f"{self.current_function_name}_{var_name}"
            if self.current_function_name not in self.func_local_cars:
                self.func_local_cars[self.current_function_name] = {}
            if var_name not in self.global_vars:
                self.global_vars[var_name] = {"new_name":unique_var_name, "node":node}
                self.func_local_cars[self.current_function_name][var_name]= unique_var_name
                node.targets[0].id = unique_var_name
            else:
                # If the variable is already defined, use the unique name
                node.targets[0].id = self.global_vars[var_name]["new_name"]
        return node

class RenameVarsInFunction(ast.NodeTransformer):
    def __init__(self, rename_map):
        # rename_map is a dictionary mapping old variable names to new ones
        self.rename_map = rename_map

    def visit_Name(self, node):
        # Rename the variable if it's in the rename map
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

def add_global_declarations(tree, global_vars):
    # Create global declarations at the beginning of the module
    for original_name in global_vars:
        unique_name = global_vars[original_name]["new_name"]
        original_node = global_vars[original_name]["node"]
        assign = ast.Assign(targets=[ast.Name(id=unique_name, ctx=ast.Store())], value=original_node.value, lineno = 0)
        tree.body.insert(0, assign)

def transform_localVar_to_globalVar(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = PromoteLocalVarsToGlobals()
    for node in ast.iter_child_nodes(root):
        if isinstance(node, ast.FunctionDef):
            transformed_tree = transformer.visit(node)
            applicable_rules.append("transform_localVar_to_globalVar")
            break

    # Add global declarations
    add_global_declarations(root, transformer.global_vars)
    
    for func in transformer.func_local_cars:
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                    rename_map = transformer.func_local_cars[func]
                    transformer = RenameVarsInFunction(rename_map)
                    transformed_tree = transformer.visit(node)

    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = transform_localVar_to_globalVar(python_code, applicable_rules)
    return update_content, applicable_rules