import ast
import random

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

class BreakInWhileLoopFinder(ast.NodeVisitor):
    def __init__(self):
        self.found_break_in_while_loop = False

    def visit_While(self, node):
        for child in ast.walk(node):
            if isinstance(child, ast.Break):
                self.found_break_in_while_loop = True
                return
            self.generic_visit(child)

        for child in node.orelse:
            self.generic_visit(child)

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

def add_loop_condition_inside(node, root):
    leftnum = random.randint(2, 1000)
    leftID = "VarLeftInWl" + str(node.lineno)
    rightnum = leftnum - 1
    rightID = "VarRightInWl" + str(node.lineno)
    leftExpr = create_assign_stat([ast.Name(id=leftID, ctx=ast.Store())], ast.Constant(value=leftnum), node.lineno)
    rightExpr = create_assign_stat([ast.Name(id=rightID, ctx=ast.Store())], ast.Constant(value=rightnum), node.lineno)
    AugAssignExpr = ast.AugAssign(target=ast.Name(id=leftID, ctx=ast.Store()), op=ast.Add(), value=ast.Constant(value=1))
    whileExpr = ast.While(test=ast.Compare(left=ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.Mod(), right=ast.Name(id=rightID, ctx=ast.Load())), ops=[ast.Eq()], comparators=[ast.Constant(value=1)]),
        body = [AugAssignExpr, node.body], lineno = node.lineno, orelse = [])
    Exprs = [leftExpr, rightExpr, whileExpr]
    node.body = Exprs

def add_nested_while_in(python_code, applicable_rules):
    root = ast.parse(python_code)
    addExprs = []
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(root)
    parent_child_dict = {}

    for node in ast.walk(root):
        if isinstance(node, ast.While):
            parent = get_parent(node, visitor)
            if parent == None:
                continue
            finder = BreakInWhileLoopFinder()
            finder.visit(node)
            Exprs_inside, Exprs_outside  = [], []
            if  finder.found_break_in_while_loop == False:
                add_loop_condition_inside(node, parent)
                applicable_rules.append("add_nested_while_in")
            break
        

    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_nested_while_in(python_code, applicable_rules)
    return update_content, applicable_rules