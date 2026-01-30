import ast
import os
import random
import utils
import shutil
import string

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

class RemoveFunctionTransformer(ast.NodeTransformer):
    def __init__(self, function_names_to_remove):
        self.function_names_to_remove = function_names_to_remove

    def visit_FunctionDef(self, node):
        if node.name in self.function_names_to_remove:
            node.body.insert(0, ast.Expr(value=ast.Str(s=f'{node.name} in newClass*.')))
            return None
        return self.generic_visit(node)
        # node.body.insert(0, ast.Expr(value=ast.Str(s=f'# {node.name} moved to a new class to create intra-dependencies.')))
        # return node

def move_functions_into_new_class(python_code, applicable_rules, target_file):
    root = ast.parse(python_code)

    imports = utils.get_imports(root)
    visitor = ImportVisitor()
    visitor.visit(root)
    import_names = visitor.imports
    funcDefs = [node for node in ast.walk(root) if isinstance(node, ast.FunctionDef)]
    idx = 0
    new_imports = []
    if len(funcDefs):
        for func_node in funcDefs:
            if idx > 0:
                break
            if "newFunc_BinOp" in func_node.name or "thread" in func_node.name or "my_decorator" in func_node.name or "dec_result" in func_node.name or func_node.decorator_list != []:
                continue
            if "return newFunc_" in ast.unparse(func_node):
                continue
            if "newFunc" not in func_node.name:
                continue
            print(ast.dump(func_node))
            new_tree = ast.Module(body=imports + [func_node], type_ignores=[])
            new_code = ast.unparse(new_tree)
            if "/" in target_file:
                dir = "/".join(target_file.split("/")[0:-1])
            else:
                dir = ""
            id = str(random.randint(2, 100000))
            newClassName = os.path.join(dir,"newClass" + id +".py")
            newModule = "newClass" + id
            utils.write_file(newClassName, new_code)
            print("Dependency Created: ", newClassName)
            os.makedirs(".tmp_test/", exist_ok=True)
            shutil.copy(newClassName, ".tmp_test/")
            new_imports = []
            if func_node.name in import_names:
                continue
            import_node = utils.create_importFrom(newModule, func_node.name, func_node.name, 0)
            # import_node = ast.ImportFrom(
            #     module= newModule,
            #     names=[ast.alias(name=func_node.name, asname=func_node.name)],
            #     level=0  # level=0 for absolute import, level=1 for relative import (e.g., .module)
            # )
            new_imports.append(import_node)
            transformer = RemoveFunctionTransformer(func_node.name)
            root = transformer.visit(root)
            applicable_rules.append("move_functions_into_new_class")
            idx += 1
            break
        root.body.insert(0,new_imports)
        # root.body.insert(0, ast.Expr(value=ast.Str(s=f'# Function were moved to a new class {newClassName}. The intra-dependencies are created correctly, but in order to check things quickly, they will be shown here in the same file.\n')))

    update_content = ast.unparse(root)
    # print(update_content)
    # exit(0)
    return update_content, applicable_rules

def init(python_code, applicable_rules, target_file):
    update_content, applicable_rules = move_functions_into_new_class(python_code, applicable_rules, target_file)
    return update_content, applicable_rules