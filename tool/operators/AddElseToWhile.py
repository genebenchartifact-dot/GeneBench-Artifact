import ast

class ASTNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parent_map = {}

    def visit(self, node, parent=None):
        """Visit a node and map it to its parent
        """
        if parent is not None:
            self.parent_map[node] = parent
        # first update the parent_map before visiting children
        super().generic_visit(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child, node)

def add_else_to_while(python_code, applicable_rules):
    root = ast.parse(python_code)
    addExprs = []
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(root)

    for node in ast.walk(root):
        if isinstance(node, ast.While):
            if node.orelse == []:
                node.orelse = ast.Pass()
                applicable_rules.append("add_else_to_while")
                break
    
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_else_to_while(python_code, applicable_rules)
    return update_content, applicable_rules