import ast
import random

"""
Better use `AddNestedForOutside` for the same functionality.
The difference between add_nested_for_outside is, 
add_nested_for_inside will not add `for` if there's `break` in original for loop.
"""

class ASTNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parent_map = {}  # Maps node with its parent

    def visit(self, node, parent=None):
        """Visit a node and map it to its parent"""
        if parent is not None:
            self.parent_map[node] = parent
        # first update the parent_map before visiting children
        super().generic_visit(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child, node)

class BreakInForLoopFinder(ast.NodeVisitor):
    def __init__(self):
        self.found_break_in_for_loop = False

    def visit_For(self, node):
        # Visit the body of the for loop
        for child in ast.walk(node):
            if isinstance(child, ast.Break):
                self.found_break_in_for_loop = True
                # No need to continue once a break is found within this for loop
                return
            # Continue searching within nested statements in the for loop's body
            self.generic_visit(child)

        # Also check in the else part of the for loop, if it exists
        for child in node.orelse:
            self.generic_visit(child)

def add_loop_condition_inside(node, root):
    leftnum = random.randint(2, 1000)
    leftID = "LoopChecker1" + str(node.lineno)  #varLeftIn_
    rightnum = leftnum - 1
    rightID = "LoopChecker2" + str(node.lineno)
    leftExpr = create_assign_stat([ast.Name(id=leftID, ctx=ast.Store())], ast.Constant(value=leftnum), node.lineno)
    rightExpr = create_assign_stat([ast.Name(id=rightID, ctx=ast.Store())], ast.Constant(value=rightnum), node.lineno)
    Exprs = [leftExpr, rightExpr]
    forExpr = ast.For(target=ast.Name(id='LoopIndex', ctx=ast.Store()), 
                iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), 
                args=[ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.Mod(), right=ast.Name(id=rightID, ctx=ast.Load()))], keywords=[]),
                body = node.body, lineno = node.lineno, orelse=[])
    node.body = forExpr
    return root, node, Exprs

def create_assign_stat(targets, value, lineno):
    return ast.Assign(targets=targets, value=value, lineno=lineno)

def get_parent(node, visitor):
    """Get the parent of the given node"""
    return visitor.parent_map.get(node, None)

def add_nested_for_in(python_code, applicable_rules):
    root = ast.parse(python_code)
    addExprs = []
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(root)

    for node in ast.walk(root):
        if isinstance(node, ast.For):
            parent = get_parent(node, visitor)
            if parent == None:
                continue
            if isinstance(parent, ast.For):
                continue
            finder = BreakInForLoopFinder()
            finder.visit(node)
            Exprs_inside  = []
            if  finder.found_break_in_for_loop == False:
                new_parent, node, Exprs_inside = add_loop_condition_inside(node, parent)
                applicable_rules.append("add_nested_for_in")

            parent_up = get_parent(node, visitor)
            under_parent_up = node
            while isinstance(parent_up, ast.If):
                under_parent_up = parent_up
                parent_up = get_parent(parent_up, visitor)
            if parent_up == None:
                parent_up = root
            try:
                idx = parent.body.index(node)
                parent.body.insert(idx, Exprs_inside)
            except:
                try:
                    idx = parent_up.body.index(under_parent_up)
                    parent_up.body.insert(idx, Exprs_inside)
                except:
                    parent_up.body.insert(0, Exprs_inside)
            break
        
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_nested_for_in(python_code, applicable_rules)
    return update_content, applicable_rules