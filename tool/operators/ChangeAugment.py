import ast

class VisitorAugAssign(ast.NodeTransformer):
    def __init__(self):
        self.counter = 0
        
    def generic_visit(self, node):
        new_node = node
        node = super().generic_visit(node)
        if isinstance(node, ast.AugAssign) and self.counter == 0:
            target_id = node.target.id
            lineno = node.lineno
            new_op = node.op 
            if isinstance(node.value, ast.Constant): # only support Constant or variable as augment value (BinOp is not supported, e.g., n -= 2 ** (int(math.log2(n))))
                increment_value = node.value.value
                newExpr = ast.Assign(targets=[ast.Name(id=target_id, ctx=ast.Store())], value=ast.BinOp(left=ast.Name(id=target_id, ctx=ast.Load()), op=new_op, right=ast.Constant(value=increment_value)), lineno = lineno)
                new_node = newExpr
                ast.copy_location(new_node, node)
                self.counter += 1
            elif isinstance(node.value, ast.Name):
                increment_value = node.value.id
                newExpr = ast.Assign(targets=[ast.Name(id=target_id, ctx=ast.Store())], value=ast.BinOp(left=ast.Name(id=target_id, ctx=ast.Load()), op=new_op, right=ast.Name(id=increment_value, ctx=ast.Load())), lineno = lineno)
                new_node = newExpr
                ast.copy_location(new_node, node)
                self.counter += 1
        return new_node

def changing_AugAssign(python_code, applicable_rules):
    """ from C += 1 to C = C +1;
    """
    root = ast.parse(python_code)
    nodeVisitor = VisitorAugAssign()
    new_root = nodeVisitor.visit(root)
    if nodeVisitor.counter > 0:
        applicable_rules.append("changing_AugAssign")
    update_content = ast.unparse(new_root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = changing_AugAssign(python_code, applicable_rules)
    return update_content, applicable_rules