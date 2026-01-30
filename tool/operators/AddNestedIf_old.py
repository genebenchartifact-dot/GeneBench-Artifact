import ast
import random

class NestedIfFinder(ast.NodeVisitor):
    def __init__(self, N):
        self.parent_map = {}  # Maps node to its parent
        self.nested_if_nodes = set()  # Stores the complete node for N-level nested if statements
        self.N = N  # Depth of nested if statements to find

    def visit(self, node, parent=None):
        """Visit a node and map it to its parent."""
        self.parent_map[node] = parent
        super().visit(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child, node)

    def find_nested_ifs(self, node, depth=0, path=[]):
        """Recursively find and return the entire if node that contains exactly N-level nested ifs."""
        if isinstance(node, ast.If):
            # Append current node to path
            path.append(node)
            # Increase depth for each if statement found
            depth += 1
            if depth == self.N:
                # If the depth is exactly N, add the highest level node in the current nested path
                if path:
                    self.nested_if_nodes.add(ast.unparse(path[0]))
            # Continue searching within the current if node
            for child in ast.iter_child_nodes(node):
                self.find_nested_ifs(child, depth, path.copy())
        else:
            # Continue searching in other types of nodes
            for child in ast.iter_child_nodes(node):
                self.find_nested_ifs(child, depth, path.copy())


def identify_nested_if_statements(code):
    tree = ast.parse(code)
    finder = NestedIfFinder(N=4)
    finder.visit(tree)  # Build the parent map
    finder.find_nested_ifs(tree)  # Find all nested ifs
    return finder.nested_if_nodes

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

class ReplaceNodeTransformer(ast.NodeTransformer):
    def __init__(self, target_node, replacement_node):
        """
        Initialize the transformer with the target node to replace and the replacement node.
        
        Args:
        target_node (ast.AST): The type of AST node you want to replace.
        replacement_node (ast.AST): The new node to replace the target node with.
        """
        self.target_node = target_node
        self.replacement_node = replacement_node

    def generic_visit(self, node):
        """
        Override the generic_visit to replace target node with the replacement node.
        """
        if isinstance(node, self.target_node.__class__):
            # if ast.dump(node) == ast.dump(self.target_node):
            if node == self.target_node:
                # print("FOUND", ast.unparse(node))
                return ast.copy_location(self.replacement_node, node)
        return super().generic_visit(node)

def get_parent(node, visitor):
    """Get the parent of the given node"""
    return visitor.parent_map.get(node, None)

def create_assign_stat(targets, value, lineno):
    return ast.Assign(targets=targets, value=value, lineno=lineno)

def add_if_outside(node, root):
    leftnum = random.randint(2, 1000)
    leftID = "ConditionChecker1" + str(node.lineno)
    rightnum = random.randint(2, 1000)
    rightID = "ConditionChecker2" + str(node.lineno)
    leftExpr = create_assign_stat([ast.Name(id=leftID, ctx=ast.Store())], ast.Constant(value=leftnum), node.lineno)
    rightExpr = create_assign_stat([ast.Name(id=rightID, ctx=ast.Store())], ast.Constant(value=rightnum), node.lineno)
    Exprs = [leftExpr, rightExpr]
    newIfExpr = ast.If(test=ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.BitAnd(), right=ast.Name(id=rightID, ctx=ast.Load())), 
        body = node, orelse=[])

    transformer = ReplaceNodeTransformer(node, newIfExpr)
    transformed_ast = transformer.visit(root)

    ast.fix_missing_locations(root)
    return root, newIfExpr, Exprs

def add_nested_if(python_code, applicable_rules):
    nested_ifs = identify_nested_if_statements(python_code)
    root = ast.parse(python_code)
    addExprs = []
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(root)
    parent_child_dict = {}
    uniq_nodes = []

    for node in ast.walk(root):
        if isinstance(node, ast.If) and ast.unparse(node) not in nested_ifs:
            parent = get_parent(node, visitor)
            if node in uniq_nodes:
                continue
            if parent in uniq_nodes:
                continue
            if isinstance(parent, ast.If): # for if-elif, only add one nested-if outside the first if;
                continue
            if "if __name__ == '__main__':" in ast.unparse(node):
                continue
            parent_up = get_parent(node, visitor)
            under_parent_up = node
            while isinstance(parent_up, ast.If) or isinstance(parent_up, ast.For):
                under_parent_up = parent_up
                parent_up = get_parent(parent_up, visitor)
            if parent_up == None:
                parent_up = root

            if node not in uniq_nodes and parent not in uniq_nodes:
                uniq_nodes.append(node)
            else:
                continue
            if parent == None:
                continue
            new_parent, node, Exprs = add_if_outside(node, parent)
            if parent_up not in parent_child_dict:
                parent_child_dict[parent_up] = {}
            if node not in parent_child_dict[parent_up]:
                parent_child_dict[parent_up][node] = {"Exprs": Exprs, "under_parent_up": under_parent_up}
            if Exprs != []:
                applicable_rules.append("add_nested_if")
            break

    for parent in parent_child_dict:
        for node in parent_child_dict[parent]:
            # idx = (parent.body).index(node)
            Exprs = parent_child_dict[parent][node]["Exprs"]
            under_parent = parent_child_dict[parent][node]["under_parent_up"]
            try:
                idx = parent.body.index(under_parent)
                parent.body.insert(idx, Exprs)
            except:
                parent.body.insert(0, Exprs)
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_nested_if(python_code, applicable_rules)
    return update_content, applicable_rules