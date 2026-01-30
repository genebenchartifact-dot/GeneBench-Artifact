import ast

class ExitCallFinder(ast.NodeVisitor):
    def __init__(self):
        self.exit_calls = []  # List to hold the nodes that are exit() calls

    def visit_Call(self, node):
        """Check if the node represents a call to exit() and record it."""
        if isinstance(node.func, ast.Name) and node.func.id == 'exit':
            self.exit_calls.append(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'quit':
            self.exit_calls.append(node)
        if ast.unparse(node) == "sys.exit()":
            self.exit_calls.append(node)
        # Important: continue traversing the tree from this node
        self.generic_visit(node)

def add_try_except_outside_functions(python_code, applicable_rules):
    root = ast.parse(python_code)
    finder = ExitCallFinder()
    # finder.visit(root)
    idx = 0
    for node in ast.iter_child_nodes(root):
        # print(node)
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.ClassDef) or isinstance(node, ast.Import) \
            or isinstance(node, ast.ImportFrom) or isinstance(node, ast.Assign) or isinstance(node, ast.Expr):
            idx += 1
            continue
        finder.visit(node)
        if finder.exit_calls:
            continue
        TryCatchNode = ast.Try(body=[node], handlers=[ast.ExceptHandler(body=[ast.Pass()])], orelse=[], finalbody=[])
        root.body.pop(idx)
        root.body.insert(idx, TryCatchNode)
        applicable_rules.append("add_try_except_outside_functions")
        idx += 1
        break
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_try_except_outside_functions(python_code, applicable_rules)
    return update_content, applicable_rules