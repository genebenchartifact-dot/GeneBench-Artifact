import ast


# This operator is to transform for loop (with range()) into recursion.
# Please note that Python itself has a limit for 1000 recursions, 
# so for huge iterations, transforming for loop into recursion will lead to runtime error.
# Such cases are not applicable for this operator.

class ForLoopTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.current_function = None  # To keep track of the current function context

    def visit_FunctionDef(self, node):
        # Keep track of the current function
        original_function = self.current_function
        self.current_function = node
        self.generic_visit(node)
        self.current_function = original_function
        return node

    def visit_For(self, node):
        self.generic_visit(node)  # First, visit all child nodes

        # print(isinstance(node.iter, ast.Call))
        # print(isinstance(node.iter.func, ast.Name))
        # print(self.counter)
        # print(node.iter.func.id)
        

        # Check if the for loop uses range()
        if isinstance(node.iter, ast.Call) and \
           isinstance(node.iter.func, ast.Name) and \
           node.iter.func.id == 'range' and self.counter < 1 and node.target.id != "_":
            # Extract range arguments
            range_args = node.iter.args
            num_args = len(range_args)
            if num_args == 1:
                start = ast.Num(n=0)
                stop = range_args[0]
                step = ast.Num(n=1)
            elif num_args == 2:
                start = range_args[0]
                stop = range_args[1]
                step = ast.Num(n=1)
            elif num_args == 3:
                start = range_args[0]
                stop = range_args[1]
                step = range_args[2]
            else:
                # Unsupported range() usage
                return node
            
            # Collect variables assigned in the loop body
            assigned_vars = self.collect_assigned_vars(node.body)
            # Remove the loop variable itself
            if isinstance(node.target, ast.Name):
                loop_var = node.target.id
                assigned_vars.discard(loop_var)
            else:
                loop_var = "_loop_var"  # Fallback loop variable name

            # Determine if variables are global or nonlocal
            scope_vars = []
            if self.current_function:
                scope_type = ast.Nonlocal
            else:
                scope_type = ast.Global
            if assigned_vars:
                scope_vars.append(scope_type(names=list(assigned_vars)))

            # Create the recursive function
            func_name = f'loop_{node.lineno}_{node.col_offset}'
            func_args = [
                ast.arg(arg=loop_var, annotation=None),
                ast.arg(arg='stop', annotation=None),
                ast.arg(arg='step', annotation=None),
            ]
            func_body = []

            # Add nonlocal/global declarations
            func_body.extend(scope_vars)

            # Add termination condition
            termination_condition = ast.If(
                test=ast.BoolOp(
                    op=ast.Or(),
                    values=[
                        ast.Compare(
                            left=ast.Name(id='step', ctx=ast.Load()),
                            ops=[ast.Eq()],
                            comparators=[ast.Num(n=0)]
                        ),
                        ast.BoolOp(
                            op=ast.And(),
                            values=[
                                ast.Compare(
                                    left=ast.Name(id='step', ctx=ast.Load()),
                                    ops=[ast.Gt()],
                                    comparators=[ast.Num(n=0)]
                                ),
                                ast.Compare(
                                    left=ast.Name(id=loop_var, ctx=ast.Load()),
                                    ops=[ast.GtE()],
                                    comparators=[ast.Name(id='stop', ctx=ast.Load())]
                                )
                            ]
                        ),
                        ast.BoolOp(
                            op=ast.And(),
                            values=[
                                ast.Compare(
                                    left=ast.Name(id='step', ctx=ast.Load()),
                                    ops=[ast.Lt()],
                                    comparators=[ast.Num(n=0)]
                                ),
                                ast.Compare(
                                    left=ast.Name(id=loop_var, ctx=ast.Load()),
                                    ops=[ast.LtE()],
                                    comparators=[ast.Name(id='stop', ctx=ast.Load())]
                                )
                            ]
                        ),
                    ]
                ),
                body=[ast.Return(value=None)],
                orelse=[]
            )
            func_body.append(termination_condition)

            # Adjust the loop body to use the correct loop variable
            # loop_body = ast.fix_missing_locations(node.body)
            func_body.extend(node.body)

            # Recursive call
            recursive_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=func_name, ctx=ast.Load()),
                    args=[
                        ast.BinOp(
                            left=ast.Name(id=loop_var, ctx=ast.Load()),
                            op=ast.Add(),
                            right=ast.Name(id='step', ctx=ast.Load())
                        ),
                        ast.Name(id='stop', ctx=ast.Load()),
                        ast.Name(id='step', ctx=ast.Load()),
                    ],
                    keywords=[]
                )
            )
            func_body.append(recursive_call)

            # Define the recursive function
            recursive_func = ast.FunctionDef(
                name=func_name,
                args=ast.arguments(
                    posonlyargs=[],
                    args=func_args,
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=func_body,
                decorator_list=[]
            )

            # Initial function call
            initial_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id=func_name, ctx=ast.Load()),
                    args=[start, stop, step],
                    keywords=[]
                )
            )
            self.counter += 1
            # Return the new nodes to replace the original for loop
            return [recursive_func, initial_call]
        else:
            return node

    def collect_assigned_vars(self, nodes):
        assigned_vars = set()
        for node in ast.walk(ast.Module(body=nodes)):
            if isinstance(node, (ast.Assign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        assigned_vars.add(target.id)
        return assigned_vars


def transform_range_to_recursion(python_code, applicable_rules):
    tree = ast.parse(python_code)
    transformer = ForLoopTransformer()
    transformer.visit(tree)
    ast.fix_missing_locations(tree)
    update_content = ast.unparse(tree)
    if transformer.counter > 0:
        applicable_rules.append("transform_range_to_recursion")
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = transform_range_to_recursion(python_code, applicable_rules)
    return update_content, applicable_rules
