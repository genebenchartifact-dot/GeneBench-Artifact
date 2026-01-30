import ast
import random

class NestedWhileLoopFinder(ast.NodeVisitor):
    def __init__(self, N):
        self.parent_map = {}  # Maps node to its parent
        self.nested_while_nodes = set()  # Stores the complete node for N-level nested while loops
        self.N = N  # Depth of nested while loops to find

    def visit(self, node, parent=None):
        """Visit a node and map it to its parent."""
        self.parent_map[node] = parent
        super().visit(node)
        for child in ast.iter_child_nodes(node):
            self.visit(child, node)

    def find_nested_while_loops(self, node, depth=0, path=[]):
        """Recursively find and return the entire while node that contains exactly N-level nested whiles."""
        if isinstance(node, ast.While):
            # Append current node to path
            path.append(node)
            # Increase depth for each while statement found
            depth += 1
            if depth >= self.N:
                # If the depth is exactly N, add the highest level node in the current nested path
                # if path:
                #     self.nested_while_nodes.add(ast.unparse(path[0]))
                for while_node in path[-self.N:]:
                    self.nested_while_nodes.add(ast.unparse(while_node))
            # Continue searching within the current while node
            for child in ast.iter_child_nodes(node):
                self.find_nested_while_loops(child, depth, path.copy())
        else:
            # Continue searching in other types of nodes
            for child in ast.iter_child_nodes(node):
                self.find_nested_while_loops(child, depth, path.copy())

def identify_nested_while_loops(code):
    tree = ast.parse(code)
    finder = NestedWhileLoopFinder(N=4)
    finder.visit(tree)  # Build the parent map
    finder.find_nested_while_loops(tree)  # Find all nested whiles
    return finder.nested_while_nodes

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

def add_loop_condition_outside(node, root):
    leftnum = random.randint(2, 1000)
    leftID = "whileloopchecker1" + str(node.lineno)
    rightnum = leftnum - 1
    rightID = "whileloopchecker2" + str(node.lineno)
    leftExpr = create_assign_stat([ast.Name(id=leftID, ctx=ast.Store())], ast.Constant(value=leftnum), node.lineno)
    rightExpr = create_assign_stat([ast.Name(id=rightID, ctx=ast.Store())], ast.Constant(value=rightnum), node.lineno)
    AugAssignExpr = ast.AugAssign(target=ast.Name(id=leftID, ctx=ast.Store()), op=ast.Add(), value=ast.Constant(value=1))
    Exprs = [leftExpr, rightExpr]
    newWhileLoop = ast.While(test=ast.Compare(left=ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.Mod(), right=ast.Name(id=rightID, ctx=ast.Load())), ops=[ast.Eq()], comparators=[ast.Constant(value=1)]),
     body=[AugAssignExpr,node], lineno = node.lineno, orelse=[])
    
    # ast.While(target=ast.Name(id='loop_outside', ctx=ast.Store()), 
    #             iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), 
    #             args=[ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.FloorDiv(), right=ast.Name(id=rightID, ctx=ast.Load()))], keywords=[]),
    #             body = node, lineno = node.lineno, orelse=[])

    transformer = ReplaceNodeTransformer(node, newWhileLoop)
    transformed_ast = transformer.visit(root)
    # index = (root.body).index(node)
    # root.body.pop(index)
    # root.body.insert(index, newWhileLoop)
    # index = (root.body).index(newWhileLoop)
    # root.body.insert(index,Exprs)
    
    ast.fix_missing_locations(root)
    return root, newWhileLoop, Exprs

def add_nested_while_out(python_code, applicable_rules):
    nested_whiles = identify_nested_while_loops(python_code)
    root = ast.parse(python_code)
    addExprs = []
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(root)
    parent_child_dict = {}

    for node in ast.walk(root):
        if isinstance(node, ast.While) and ast.unparse(node) not in nested_whiles:
            parent = get_parent(node, visitor)
            if parent == None:
                continue
            finder = BreakInWhileLoopFinder()
            finder.visit(node)
            Exprs_inside, Exprs_outside  = [], []
            # if  finder.found_break_in_while_loop == False:
            #     new_parent, node, Exprs_inside  = add_loop_condition_inside(node, parent)
            new_parent, node, Exprs_outside = add_loop_condition_outside(node, parent)
            applicable_rules.append("add_nested_while_out")

            under_parent_up = node
            parent_up = get_parent(node, visitor)
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

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_nested_while_out(python_code, applicable_rules)
    return update_content, applicable_rules