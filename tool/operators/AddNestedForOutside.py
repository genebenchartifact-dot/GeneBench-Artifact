import ast
import random

class NestedForLoopFinder(ast.NodeVisitor):
    def __init__(self, N):
        self.parent_map = {}  # Maps node to its parent
        self.nested_loops = set()  # Stores the complete nodes for N-level nested for loops
        self.N = N  # Depth of nested for loops to find

    def visit(self, node, parent=None):
        """Visit a node and map it to its parent."""
        self.parent_map[node] = parent
        super().visit(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child, node)

    def find_nested_for_loops(self, node, depth=0, path=[]):
        """Recursively find and return the entire for node that contains exactly N-level nested fors."""
        if isinstance(node, ast.For):
            # Append current node to path
            path.append(node)
            # Increase depth for each for statement found
            depth += 1
            if depth >= self.N:
                # Add all for loops in the path that are part of the N-level nested structure
                for for_node in path[-self.N:]:
                    self.nested_loops.add(ast.unparse(for_node))
            # Continue searching within the current for node
            for child in ast.iter_child_nodes(node):
                self.find_nested_for_loops(child, depth, path.copy())
        else:
            # Continue searching in other types of nodes
            for child in ast.iter_child_nodes(node):
                self.find_nested_for_loops(child, depth, path.copy())

def identify_nested_for_loops(code):
    tree = ast.parse(code)
    finder = NestedForLoopFinder(N=4)
    finder.visit(tree)  # Build the parent map
    finder.find_nested_for_loops(tree)  # Find all nested loops
    return finder.nested_loops


class ASTNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parent_map = {}  # Maps node to its parent

    def visit(self, node, parent=None):
        """Visit a node and map it to its parent"""
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

def add_loop_condition_outside(node, root):
    leftnum = random.randint(2, 1000)
    leftID = "LoopChecker1" + str(node.lineno)
    rightnum = leftnum - 1
    rightID = "LoopChecker2" + str(node.lineno)
    leftExpr = create_assign_stat([ast.Name(id=leftID, ctx=ast.Store())], ast.Constant(value=leftnum), node.lineno)
    rightExpr = create_assign_stat([ast.Name(id=rightID, ctx=ast.Store())], ast.Constant(value=rightnum), node.lineno)
    Exprs = [leftExpr, rightExpr]
    newForLoop = ast.For(target=ast.Name(id='LoopIndexOut', ctx=ast.Store()), 
                iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), 
                args=[ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.FloorDiv(), right=ast.Name(id=rightID, ctx=ast.Load()))], keywords=[]),
                body = node, lineno = node.lineno, orelse=[])

    transformer = ReplaceNodeTransformer(node, newForLoop)
    transformed_ast = transformer.visit(root)
    # index = (root.body).index(node)
    # root.body.pop(index)
    # root.body.insert(index, newForLoop)
    # index = (root.body).index(newForLoop)
    # root.body.insert(index,Exprs)
    
    ast.fix_missing_locations(root)
    return root, newForLoop, Exprs

def add_nested_for_out(python_code, applicable_rules):
    nested_loops = identify_nested_for_loops(python_code)
    root = ast.parse(python_code)
    addExprs = []
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(root)
    for node in ast.walk(root):
        if isinstance(node, ast.For) and ast.unparse(node) not in nested_loops:
            parent = get_parent(node, visitor)
            if parent == None:
                continue
            if isinstance(parent, ast.For):
                continue
            Exprs_outside  = []
            new_parent, node, Exprs_outside = add_loop_condition_outside(node, parent)
            applicable_rules.append("add_nested_for_out")

            parent_up = get_parent(node, visitor)
            under_parent_up = node
            while isinstance(parent_up, ast.If):
                under_parent_up = parent_up
                parent_up = get_parent(parent_up, visitor)
            if parent_up == None:
                parent_up = root
            try:
                idx = parent.body.index(node)
                parent.body.insert(idx, Exprs_outside)
            except:
                try:
                    idx = parent_up.body.index(under_parent_up)
                    parent_up.body.insert(idx, Exprs_outside)
                except:
                    parent_up.body.insert(0, Exprs_outside)
                    
            break
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def create_assign_stat(targets, value, lineno):
    return ast.Assign(targets=targets, value=value, lineno=lineno)

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_nested_for_out(python_code, applicable_rules)
    return update_content, applicable_rules