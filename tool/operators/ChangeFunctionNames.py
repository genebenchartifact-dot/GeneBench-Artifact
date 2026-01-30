import ast

class RenameFunctionTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.function_count = 0
        self.name_map = {}
        self.in_class = False

    def visit_ClassDef(self, node):
        self.in_class = True
        self.generic_visit(node)  # Visit all children nodes
        self.in_class = False
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.function_count < 1 and "Func_" not in node.name and node.name != "__init__" and \
            node.name != "dec_result" and node.name != "my_decorator" and not self.in_class:
            original_name = node.name
            new_name = f"Func_{original_name}_{self.function_count}"
            self.name_map[original_name] = new_name
            node.name = new_name
            self.function_count += 1
            self.generic_visit(node)
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.name_map:
                node.func.id = self.name_map[node.func.id]
        self.generic_visit(node)
        return node

def change_function_names(python_code, applicable_rules):
    root = ast.parse(python_code)
    transformer = RenameFunctionTransformer()
    new_tree = transformer.visit(root)
    if transformer.function_count > 0:
        applicable_rules.append("change_function_names")
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = change_function_names(python_code, applicable_rules)
    return update_content, applicable_rules