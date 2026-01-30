import ast
import astor
import random

class ReplaceNodeTransformer(ast.NodeTransformer):
    def __init__(self, target_node, replacement_node):
        self.target_node = target_node
        self.replacement_node = replacement_node

    def generic_visit(self, node):
        if isinstance(node, self.target_node.__class__):
            # if ast.dump(node) == ast.dump(self.target_node):
            if node == self.target_node:
                # print("FOUND", ast.unparse(node))
                return ast.copy_location(self.replacement_node, node)
        return super().generic_visit(node)

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

class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()  # Use a set to avoid duplicates

    def visit_Name(self, node):
        self.variables.add(node)
        self.generic_visit(node)

def get_variables_from_binop(binop_node):
    collector = VariableCollector()
    collector.visit(binop_node)
    return collector.variables

def get_parent(node, visitor):
    return visitor.parent_map.get(node, None)

def ListToVar(node, root):

    newExprs = []
    idx = get_index(node, root) #(root.body).index(node)
    if idx == None:
        raise "Error in ListToVar, cannot locate node in parent node"

    name_pairs = {}
    name_idx = 0
    # isinstance(sub, ast.Constant)
    for sub in ast.walk(node.value):
        if isinstance(sub, ast.List) or isinstance(sub, ast.Tuple) or isinstance(sub, ast.ListComp) \
        or isinstance(sub, ast.Dict) or isinstance(sub, ast.Set) or isinstance(sub, ast.SetComp) or isinstance(sub, ast.DictComp):
            new_id = "variable" + "_" + str(name_idx) + "_" + str(sub.lineno)
            newExpr = ast.Assign(targets=[ast.Name(id=new_id, ctx=ast.Store())], value=sub, lineno = sub.lineno)
            loadExpr = ast.Name(id=new_id, ctx=ast.Load())
            if sub not in name_pairs:
                name_pairs[sub] = loadExpr
            newExprs.append(newExpr)
        name_idx += 1

    if name_pairs == {}:
        return root, newExprs

    for child in ast.walk(node):
        if child in name_pairs:
            transformer = ReplaceNodeTransformer(child, name_pairs[child])
            transformed_ast = transformer.visit(node)
    return root, newExprs

def OthersToVar(node, root):

    newExprs = []
    idx = get_index(node, root) #(root.body).index(node)
    if idx == None:
        raise "Error in OthersToVar, cannot locate node in parent node"

    name_pairs = {}
    name_idx = 0
    # isinstance(sub, ast.Constant)
    for sub in ast.walk(node.value):
        if isinstance(sub, ast.Subscript) or isinstance(sub, ast.Starred): # or isinstance(sub, ast.Attribute) or isinstance(sub, ast.Starred): # or isinstance(sub, ast.List) or isinstance(sub, ast.Tuple):
            new_id = "variable" + "_" + str(name_idx) + "_" + str(sub.lineno)
            newExpr = ast.Assign(targets=[ast.Name(id=new_id, ctx=ast.Store())], value=sub, lineno = sub.lineno)
            loadExpr = ast.Name(id=new_id, ctx=ast.Load())
            if sub not in name_pairs:
                name_pairs[sub] = loadExpr
            newExprs.append(newExpr)
        name_idx += 1

    if name_pairs == {}:
        return root, newExprs

    for child in ast.walk(node):
        if child in name_pairs:
            transformer = ReplaceNodeTransformer(child, name_pairs[child])
            transformed_ast = transformer.visit(node)
    return root, newExprs

def ConstToVar(node, root):
    newExprs = []
    idx = get_index(node, root)
    if idx == None:
        raise "Error in OthersToVar, cannot locate node in parent node"

    name_pairs = {}
    name_idx = 0
    # isinstance(sub, ast.Constant)
    for sub in ast.walk(node.value):
        if isinstance(sub, ast.Constant):
            new_id = "variable" + "_" + str(name_idx) + "_" + str(sub.lineno)
            newExpr = ast.Assign(targets=[ast.Name(id=new_id, ctx=ast.Store())], value=sub, lineno = sub.lineno)
            loadExpr = ast.Name(id=new_id, ctx=ast.Load())
            if sub not in name_pairs:
                name_pairs[sub] = loadExpr
            newExprs.append(newExpr)
        name_idx += 1

    if name_pairs == {}:
        return root, newExprs

    # new_idx = get_index(node, root)
    # for child in ast.walk(node):
    #     if child in name_pairs:
    #         doubler = ReplaceNode(child, name_pairs[child])
    #         new_tree = doubler.visit(node)
    
    for child in ast.walk(node):
        if child in name_pairs:
            transformer = ReplaceNodeTransformer(child, name_pairs[child])
            transformed_ast = transformer.visit(node)
    # print(ast.unparse(root))
    # print(new_idx)
    # root.body.pop(new_idx)
    # root.body.insert(new_idx, new_tree)
    # print(ast.unparse(root))
    return root, newExprs

def get_index(child_node, parent_node):
    idx = 0
    for node in ast.iter_child_nodes(parent_node):
        if ast.dump(node) == ast.dump(child_node):
            return idx
        idx += 1
    return None

def create_functions(python_code, applicable_rules):
    root = ast.parse(python_code)
    to_remove_consts = []

    visitor = ASTNodeVisitor()
    visitor.visit(root)
    parent_child_dict = {}

    """convert all objects to variable (ast.Name)"""
    func_name_idx = 0
    for node in ast.walk(root):
        if isinstance(node, ast.Assign) or isinstance(node, ast.AugAssign):
            # print(ast.dump(node))
            if isinstance(node.value, ast.BinOp):
                parent = get_parent(node, visitor)
                parent_up = get_parent(node, visitor)
                under_parent_up = node

                while isinstance(parent_up, ast.If):
                    under_parent_up = parent_up
                    parent_up = get_parent(parent_up, visitor)
                if parent_up == None:
                    parent_up = root
                new_parent, newExprsList =  ListToVar(node, parent)
                new_parent, newExprsOthers =  OthersToVar(node, parent)
                new_parent, newExprsConst =  ConstToVar(node, parent)

                allExprs = newExprsList + newExprsOthers + newExprsConst
                try:
                    idx = parent.body.index(node)
                    parent.body.insert(idx, allExprs)
                except:
                    try:
                        idx = parent_up.body.index(under_parent_up)
                        parent_up.body.insert(idx, allExprs)
                    except:
                        parent_up.body.insert(0, allExprs)

                """extract functions"""
                vars = get_variables_from_binop(node.value)
                args_id = []
                args_node = []
                actual_args = []
                for var in vars:
                    if var.id not in actual_args:
                        args_id.append(ast.arg(arg=var.id))
                        actual_args.append(var.id)
                        args_node.append(ast.Name(id=var.id, ctx=ast.Load()))
                
                lineno = node.lineno
                newFunc = ast.FunctionDef(name="newFunc" + str(func_name_idx) + "_" + str(lineno), args=ast.arguments(posonlyargs=[], 
                        args=args_id, kwonlyargs=[], kw_defaults=[], defaults=[]), 
                        body=[ast.Return(value=node.value)], 
                        decorator_list=[], lineno = lineno)
                if isinstance(node, ast.Assign):
                    newExpr = ast.Assign(targets=node.targets, 
                            value=ast.Call(func=ast.Name(id=newFunc.name, ctx=ast.Load()), 
                            args=args_node, 
                            keywords=[]), lineno = lineno)
                elif isinstance(node, ast.AugAssign):
                    newExpr = ast.AugAssign(target=node.target, op = node.op,
                            value=ast.Call(func=ast.Name(id=newFunc.name, ctx=ast.Load()), 
                            args=args_node, 
                            keywords=[]), lineno = lineno)
                root.body.insert(0, newFunc)
                transformer = ReplaceNodeTransformer(node, newExpr)
                transformed_ast = transformer.visit(parent)
                applicable_rules.append("create_functions")
                """
                # node = newExpr
                transformer = ReplaceNodeTransformer(node, newExpr)
                transformed_ast = transformer.visit(parent)
                # print(ast.unparse(root))
                # idx = (parent.body).index(node)
                # parent.body.pop(idx)
                # parent.body.insert(idx, newExpr)
                root.body.insert(0, newFunc)
                """
                func_name_idx += 1
                break

    update_content = ast.unparse(root)
    # print(update_content)
    # exit(0)
    return update_content, applicable_rules

class ReturnNodeTransformer(ast.NodeTransformer):
    def __init__(self):
        self.new_functions = [] 
        self.counter = 0

    def visit_Return(self, node):
        if self.counter > 1:
            return node
        if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = node.value.left
            right = node.value.right
            op = node.value.op
            
            func_name = f"newFunc_{random.randint(1000, 100000)}"
            left_arg = ast.Name(id=f"arg{self.counter}", ctx=ast.Store()) 
            right_arg = ast.Name(id=f"arg{self.counter+1}", ctx=ast.Store()) 
            
            new_function = ast.FunctionDef(
                name=func_name,
                args=ast.arguments(
                    args=[
                        ast.arg(arg=astor.to_source(left_arg).strip()),
                        ast.arg(arg=astor.to_source(right_arg).strip())
                    ],
                    posonlyargs=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=[ast.Return(value=ast.BinOp(left=left_arg, op=op, right=right_arg))],
                decorator_list=[],
                lineno = node.lineno
            )

            new_call = ast.Call(
                func=ast.Name(id=func_name, ctx=ast.Load()),
                args=[left, right],
                keywords=[]
            )
            node.value = new_call
            print(ast.unparse(new_function))

            self.new_functions.append(new_function)
            self.counter += 1

        return node

    def visit_Module(self, node):
        self.generic_visit(node)  
        node.body = self.new_functions + node.body
        return node
    
def create_functions_for_return(python_code, applicable_rules):
    tree = ast.parse(python_code)
    transformer = ReturnNodeTransformer()
    tree = transformer.visit(tree)
    if transformer.counter > 0:
        applicable_rules.append("create_functions")
    update_content = ast.unparse(tree)
    return update_content, applicable_rules

class BinaryOpTransformer(ast.NodeTransformer):
    def __init__(self):
        self.function_count = 0

    def visit_BinOp(self, node):
        self.generic_visit(node)
        
        if self.function_count > 0:
            return node

        func_name = f"newFunc_BinOp{self.function_count}"
        self.function_count += 1
        
        new_func = ast.FunctionDef(
            name=func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg='a', annotation=None), ast.arg(arg='b', annotation=None)],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[]
            ),
            body=[ast.Return(value=ast.BinOp(left=ast.Name(id='a', ctx=ast.Load()), op=node.op, right=ast.Name(id='b', ctx=ast.Load())))],
            decorator_list=[],
            returns=None,
            lineno = node.lineno
        )

        self.functions.append(new_func)

        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[node.left, node.right],
            keywords=[]
        )

    def visit_Module(self, node):
        self.functions = []  
        self.generic_visit(node)  
        node.body = self.functions + node.body 
        return node


def create_functions_for_random_binary(python_code, applicable_rules):
    
    tree = ast.parse(python_code)
    transformer = BinaryOpTransformer()
    tree = transformer.visit(tree)
    if transformer.function_count > 0:
        applicable_rules.append("create_functions")
    update_content = ast.unparse(tree)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = create_functions(python_code, applicable_rules)
    if "create_functions" not in applicable_rules:
        update_content, applicable_rules = create_functions_for_return(python_code, applicable_rules)
    if "create_functions" not in applicable_rules:
        update_content, applicable_rules = create_functions_for_random_binary(python_code, applicable_rules)
    return update_content, applicable_rules