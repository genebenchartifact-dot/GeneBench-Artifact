import ast
import random

class IfNestingAnalyzer(ast.NodeVisitor):
    """ Analyze Python AST to find if-statements with less than specified depth of nesting. """
    def __init__(self, max_depth):
        self.max_depth = max_depth  # Maximum depth of nested if-statements
        self.parent_map = {}  # Maps node to its parent node
        self.targets = []  # Collects if-nodes with less than max_depth

    def visit_If(self, node):
        depth = self._get_depth(node)
        if depth < self.max_depth:
            self.targets.append(node)
        self.generic_visit(node)

    def _get_depth(self, node):
        depth = 0
        current = node
        while current in self.parent_map:
            if isinstance(current, ast.If):
                depth += 1
            current = self.parent_map[current]
        return depth

    def generic_visit(self, node):
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        super().generic_visit(node)
        
def create_assign_stat(targets, value, lineno):
    return ast.Assign(targets=targets, value=value, lineno=lineno)

class IfAdder(ast.NodeTransformer):
    """ AST Transformer to add a new nested if-statement outside targeted if-statements. """
    def __init__(self, targets, visitor, root, parent_child_dict):
        self.targets = targets  # if-nodes where new nesting will be added
        self.counter = 0
        self.visitor = visitor
        self.root = root
        self.parent_child_dict = parent_child_dict

    def visit_If(self, node):
        self.generic_visit(node)
        if "if __name__ == '__main__':" in ast.unparse(node):
            return node
        if node in self.targets and self.counter < 1:
            leftnum = random.randint(2, 1000)
            leftID = "ConditionChecker1" + str(node.lineno)
            rightnum = random.randint(2, 1000)
            rightID = "ConditionChecker2" + str(node.lineno)
            leftExpr = create_assign_stat([ast.Name(id=leftID, ctx=ast.Store())], ast.Constant(value=leftnum), node.lineno)
            rightExpr = create_assign_stat([ast.Name(id=rightID, ctx=ast.Store())], ast.Constant(value=rightnum), node.lineno)
            Exprs = [leftExpr, rightExpr]
            newIfExpr = ast.If(test=ast.BinOp(left=ast.Name(id=leftID, ctx=ast.Load()), op=ast.BitAnd(), right=ast.Name(id=rightID, ctx=ast.Load())), 
                body = node, orelse=[])
            # condition = ast.parse('random.randint(1, 100) > 50').body[0].value
            # new_if = ast.If(test=condition, body=[node], orelse=[])
            parent = get_parent(node, self.visitor)
            # if isinstance(parent, ast.If): # for if-elif, only add one nested-if outside the first if;
            #     return node
            parent_up = get_parent(node, self.visitor)
            under_parent_up = node
            while isinstance(parent_up, ast.If) or isinstance(parent_up, ast.For):
                under_parent_up = parent_up
                parent_up = get_parent(parent_up, self.visitor)
            if parent_up == None:
                parent_up = self.root
                
            if parent == None:
                return node

            if parent_up not in self.parent_child_dict:
                self.parent_child_dict[parent_up] = {}
            if node not in self.parent_child_dict[parent_up]:
                self.parent_child_dict[parent_up][node] = {"Exprs": Exprs, "under_parent_up": under_parent_up}
            self.counter += 1
            return ast.copy_location(newIfExpr, node)
        return node

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
            
def get_parent(node, visitor):
    """Get the parent of the given node"""
    return visitor.parent_map.get(node, None)

def add_nested_if(code, max_depth):
    tree = ast.parse(code)
    analyzer = IfNestingAnalyzer(max_depth)
    analyzer.visit(tree)
    idx = []
    visitor = ASTNodeVisitor()
    visitor.visit(tree)
    parent_child_dict = {}

    if not analyzer.targets:
        return code  # No modification needed if no targets found

    transformer = IfAdder(analyzer.targets, visitor, tree, parent_child_dict)
    tree = transformer.visit(tree)
    
    for parent in transformer.parent_child_dict:
        for node in transformer.parent_child_dict[parent]:
            # idx = (parent.body).index(node)
            Exprs = transformer.parent_child_dict[parent][node]["Exprs"]
            under_parent = transformer.parent_child_dict[parent][node]["under_parent_up"]
            try:
                idx = parent.body.index(under_parent)
                parent.body.insert(idx, Exprs)
                # print(ast.unparse(parent))
            except:
                parent.body.insert(0, Exprs)

    return ast.unparse(tree), transformer.counter


def init(python_code, applicable_rules):

    update_content, counter = add_nested_if(python_code, max_depth=5)
    if counter >0:
        applicable_rules.append("add_nested_if")
    # update_content, applicable_rules = add_nested_if(python_code, applicable_rules)
    return update_content, applicable_rules