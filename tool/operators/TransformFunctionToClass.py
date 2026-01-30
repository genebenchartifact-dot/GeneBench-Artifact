import ast
import random

class TransformToClass(ast.NodeTransformer):
    def __init__(self):
        self.class_definition = ast.ClassDef(
            name="newClass" + str(random.randint(1, 10000)),
            bases=[],
            keywords=[],
            body=[],
            decorator_list=[]
        )
        self.tochange = []
        self.count = 0

    def visit_Module(self, node):
        self.generic_visit(node)
        node.body = [self.class_definition] + node.body
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.count > 0:
            return node
        if "newFunc_BinOp" in node.name or "thread" in node.name or "my_decorator" in node.name or "dec_result" in node.name or node.decorator_list != []:
            return node
        node.args.args.insert(0, ast.arg(arg='self', annotation=None))
        self.class_definition.body.append(node)
        self.count += 1
        self.tochange.append(node.name)
        return None

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.tochange:
            func_name = node.func.id

            class_instance = ast.Call(func=ast.Name(id=self.class_definition.name, ctx=ast.Load()), args=[], keywords=[])
            method_call = ast.Call(
                func=ast.Attribute(value=class_instance, attr=func_name, ctx=ast.Load()),
                args=node.args,
                keywords=node.keywords
            )

            return ast.copy_location(method_call, node)

        return node

def transform_function_to_class(python_code, applicable_rules):
    tree = ast.parse(python_code)
    transformer = TransformToClass()
    new_tree = transformer.visit(tree)
    # print(ast.dump(new_tree))
    if transformer.count > 0:
        applicable_rules.append("transform_function_to_class")
    new_source_code = ast.unparse(new_tree)
    # print(new_source_code)
    return new_source_code, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = transform_function_to_class(python_code, applicable_rules)
    return update_content, applicable_rules