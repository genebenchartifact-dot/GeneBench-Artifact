import ast
import os
import utils

class CustomFunctionCollector(ast.NodeVisitor):
    def __init__(self):
        self.custom_functions = set()

    def visit_FunctionDef(self, node):
        self.custom_functions.add(node.name)
        self.generic_visit(node)

class FunctionTransformer(ast.NodeTransformer):
    def __init__(self, custom_functions):
        self.custom_functions = custom_functions
        self.counter = 0
        
    def visit_Assign(self, n):
        target = n.targets[0]
        if isinstance(n.value, ast.Call):
            node = n.value
            print(ast.dump(node))
            if isinstance(node.func, ast.Name) and node.func.id in self.custom_functions \
                and self.counter < 1 and "{}_thread".format(node.func.id) not in self.custom_functions:
                func_name = node.func.id

                queue_name = f"queue_{func_name}{self.counter}"
                thread_name = f"thread_{func_name}{self.counter}"
                result_name = f"result_{func_name}{self.counter}"

                queue_init = ast.Assign(
                    targets=[ast.Name(id=queue_name, ctx=ast.Store())],
                    value=ast.Call(func=ast.Name(id='queue.Queue', ctx=ast.Load()), args=[], keywords=[]),
                    lineno = node.lineno
                )
                
                def_worker = ast.FunctionDef(
                    name=f"{func_name}_thread",
                    args=ast.arguments(
                        args=[ast.arg(arg='queue', annotation=None)],
                        vararg=None,
                        kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs = []
                    ),
                    body=[
                        ast.Assign(
                            targets=[ast.Name(id='result', ctx=ast.Store())],
                            value=ast.Call(func=ast.Name(id=func_name, ctx=ast.Load()), args=node.args, keywords=node.keywords),
                            lineno = node.lineno
                        ),
                        ast.Expr(
                            value=ast.Call(
                                func=ast.Attribute(
                                    value=ast.Name(id='queue', ctx=ast.Load()),
                                    attr='put',
                                    ctx=ast.Load()
                                ),
                                args=[ast.Name(id='result', ctx=ast.Load())],
                                keywords=[]
                            )
                        )
                    ],
                    decorator_list=[],
                    lineno = node.lineno
                )

                thread_init = ast.Assign(
                    targets=[ast.Name(id=thread_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Name(id='threading.Thread', ctx=ast.Load()),
                        args=[
                            ast.keyword(arg='target', value=ast.Name(id=f'{func_name}_thread', ctx=ast.Load())),
                            ast.keyword(arg='args', value=ast.Tuple(elts=[ast.Name(id=queue_name, ctx=ast.Load())], ctx=ast.Load()))
                        ],
                        keywords=[]
                    ),
                    lineno = node.lineno
                )

                start_thread = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=thread_name, ctx=ast.Load()),
                            attr='start',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    )
                )

                join_thread = ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=thread_name, ctx=ast.Load()),
                            attr='join',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    )
                )

                get_result = ast.Assign(
                    targets=[ast.Name(id=result_name, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=queue_name, ctx=ast.Load()),
                            attr='get',
                            ctx=ast.Load()
                        ),
                        args=[],
                        keywords=[]
                    ),
                    lineno = node.lineno
                )
                self.counter += 1
                # for n in [queue_init, def_worker, thread_init, start_thread, join_thread, get_result, ast.Name(id=result_name, ctx=ast.Load())]:
                #     print(ast.unparse(n))
                # print(ast.unparse(node))
                # exit(0)
                # if self.current_assignment:
                assign_result = ast.Assign(
                    targets=[target],
                    value=ast.Name(id=result_name, ctx=ast.Load()),
                    lineno = node.lineno
                )
                self.current_assignment = None
                return [queue_init, def_worker, thread_init, start_thread, join_thread, get_result, assign_result]
                # else:
                #     return [queue_init, def_worker, thread_init, start_thread, join_thread, get_result, ast.Name(id=result_name, ctx=ast.Load())]

        return self.generic_visit(n)
    
def add_imports(root):
    """
    import threading
    import time
    Add them to root node, and return an updated root.
    """
    threadingImport = utils.create_import("threading", None)
    timeImport = utils.create_import("time", None)
    timeImport = utils.create_import("queue", None)
    root.body.insert(0, [threadingImport,timeImport])
    return root

def add_thread(python_code, applicable_rules):
    root = ast.parse(python_code)
    
    collector = CustomFunctionCollector()
    collector.visit(root)
    
    transformer = FunctionTransformer(collector.custom_functions)
    transformed_tree = transformer.visit(root)
    if transformer.counter > 0:
        root = add_imports(root)
        applicable_rules.append("add_thread")

    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = add_thread(python_code, applicable_rules)
    return update_content, applicable_rules