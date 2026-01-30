import ast

class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name:
                self.imports.append(alias.name)
            if alias.asname:
                self.imports.append(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name:
                self.imports.append(alias.name)
            if alias.asname:
                self.imports.append(alias.asname)
        self.generic_visit(node)

class ReplaceNumpy(ast.NodeTransformer):
    def __init__(self):
        self.funcs = []
    def visit_Call(self, node):
        funcs = {
            "max": "max",
            "min": "min",
            "sum": "sum",
            "abs": "abs",
            "sorted": "sort",
            "reversed": "flip"
        }
        self.generic_visit(node)  # First visit the children of the node

        # Check if the function called is 'max' or 'min'
        if isinstance(node.func, ast.Name) and node.func.id in funcs and len(self.funcs) == 0:
            # Determine which NumPy function to use
            numpy_func = funcs[node.func.id]

            # Wrap the arguments in a np.array() call
            np_array_call = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='np', ctx=ast.Load()),
                    attr='array',
                    ctx=ast.Load()),
                args=[ast.List(elts=node.args, ctx=ast.Load())],
                keywords=[]
            )

            # Replace the original function call
            node.func = ast.Attribute(
                value=ast.Name(id='np', ctx=ast.Load()),
                attr=numpy_func,
                ctx=ast.Load()
            )
            node.args = [np_array_call]  # Replace args with a single np.array call
            self.funcs.append(node.func)

        return node

def replace_with_numpy(python_code, applicable_rules):
    root = ast.parse(python_code)
    visitor = ImportVisitor()
    visitor.visit(root)
    import_names = visitor.imports
    numpy_import = ast.Import(names=[ast.alias(name="numpy", asname="np")])
    transformer = ReplaceNumpy()
    modified_ast = transformer.visit(root)
    if len(transformer.funcs):
        if "np" not in import_names:
            root.body.insert(0, numpy_import)
        applicable_rules.append("replace_with_numpy")
    update_content = ast.unparse(root)
    return update_content, applicable_rules

def init(python_code, applicable_rules):
    update_content, applicable_rules = replace_with_numpy(python_code, applicable_rules)
    return update_content, applicable_rules