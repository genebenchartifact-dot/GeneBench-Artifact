import ast
import contextlib
import os
import io
import sys
import signal
import time
import utils
import itertools
import subprocess
import tempfile
# from pycfg.pycfg import PyCFG, CFGNode, slurp

def execute_code_and_capture_output(code, inputs, filename="temp_script.py", timeout_duration=10):
    with open(filename, 'w') as file:
        file.write(code)
    try:
        result = subprocess.run(
            ['python', filename],
            input=inputs,
            capture_output=True,
            text=True,
            timeout=timeout_duration
        )
        output = result.stdout
        errors = result.stderr
    except subprocess.TimeoutExpired:
        output, errors = None, "Process timed out"
    finally:
        os.remove(filename)

    return output, errors

def code_compile_with_inputs(code, input_values):
    inputs = [item for item in input_values.split("\n") if item]
    output = None
    try:
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {e}", None, None

    try:
        input_iterator = iter(inputs)
        def custom_input(prompt=''):
            return next(input_iterator)
        
        local_context = {'input': custom_input}
        exec(compiled_code, {}, local_context)

    except Exception as e:
        return False, f"Runtime error: {e}", None, None

    output, errors = execute_code_and_capture_output(code, input_values) #new_stdout.getvalue()
    return True, "Code executed with inputs successfully", local_context, output

def code_compile_without_inputs(code):
    try:
        # Attempt to compile the code
        compiled_code = compile(code, '<string>', 'exec')
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    
    return True, "Code compiled successfully"

def extract_nodes(source_code):
    tree = ast.parse(source_code)
    nodes = [node for node in ast.iter_child_nodes(tree)]
    return nodes

def generate_permutations(nodes):
    subsets = []
    for r in range(1, len(nodes) + 1):
        subsets.extend(itertools.combinations(nodes, r))
    return subsets

def test_permutations_without_input(source_code, beginning = None, max = 5):
    nodes = extract_nodes(source_code)
    node_permutations = generate_permutations(nodes)
    compilable_permutations = []

    for permutation in node_permutations:
        code = f"{beginning}\n" + '\n'.join(ast.unparse(node) for node in permutation)
        result, message = code_compile_without_inputs(code)
        if result:
            compilable_permutations.append([node for node in permutation])
        if len(compilable_permutations) > max:
            break

    return compilable_permutations

def combine_and_test(compilable1, compilable2, test_input, beginning_code):
    combined_compilable = []
    for subset1 in compilable1:
        code1 = f"{beginning_code}\n" + '\n'.join(ast.unparse(node) for node in subset1)
        for subset2 in compilable2:
            code2 = '\n'.join(ast.unparse(node) for node in subset2)
            combined_code = code1 + "\n" + code2
            result, message = code_compile_without_inputs(combined_code)
            # result, message, inputs, outputs = code_compile_with_inputs(combined_code, test_input)
            # print(combined_code, "00")
            # print(result, message, inputs, outputs)
            if result:
                # print("\n==========START===========")
                # print(f"combined_code:{combined_code}")
                # print("==========RESULT===========")
                # print(f"successfully executed: {result}\nmessage:{message}\noutputs:{outputs}")
                combined_compilable.append(combined_code)
                # print("==========END===========")
    return combined_compilable

class InputCallFinder(ast.NodeVisitor):
    def __init__(self):
        self.all_nodes = []  # List to store all statements
        self.last_input_index = -1  # Index of the last statement containing 'input()'

    def generic_visit(self, node):
        # Add each node to the all_nodes list if it is a statement
        if isinstance(node, ast.stmt):
            self.all_nodes.append(node)
        super().generic_visit(node)

    def visit_Call(self, node):
        # Check if this node is a call to 'input'
        if isinstance(node.func, ast.Name) and node.func.id == 'input':
            # Update the index of the last input statement
            self.last_input_index = len(self.all_nodes) - 1
        # Continue traversal with generic visit
        super().generic_visit(node)
        
class InputDependencyFinder(ast.NodeVisitor):
    def __init__(self):
        self.dependencies = []  # This will store all dependent nodes as strings

    def visit_Call(self, node):
        # Check if the call is to the 'input' function
        if isinstance(node.func, ast.Name) and node.func.id == 'input':
            # Store the input call itself
            self.dependencies.append(node)
            # Visit parent nodes to find assignments or other dependencies
            self.visit_parents(node)
        self.generic_visit(node)

    def visit_parents(self, node):
        """ Recursively visit parent nodes to collect dependencies. """
        parent = getattr(node, 'parent', None)
        if parent is None:
            return
        if isinstance(parent, ast.AST):
            if not isinstance(parent, ast.Module):  # Ignore the module itself
                # Convert parent node to source code string and add to dependencies
                self.dependencies.append(parent)
            self.visit_parents(parent)

    def assign_parents(self, node, parent=None):
        """ Assign parent nodes to each node in the AST for backtracking dependencies. """
        for child in ast.iter_child_nodes(node):
            setattr(child, 'parent', node)
            self.assign_parents(child, node)

def extract_input_nodes(program):
    tree = ast.parse(program)
    finder = InputDependencyFinder()
    finder.assign_parents(tree)  # Assign parent nodes
    finder.visit(tree)
    return "\n".join([ast.unparse(stat) for stat in finder.dependencies if isinstance(stat, (ast.FunctionDef,
                                                                                   ast.Assign))])

def remove_lines_from_code(source_code, lines_to_remove):
    lines = source_code.splitlines()
    filtered_lines = [line for line in lines if line.strip() not in lines_to_remove]
    new_source_code = '\n'.join(filtered_lines)
    return new_source_code    

def clean_code(file):
    code = utils.read_file(file)
    return ast.unparse(ast.parse(code))

def extract_import_nodes(source_code):
    tree = ast.parse(source_code)
    imports = [] 

    class ImportCollector(ast.NodeVisitor):
        def visit_Import(self, node):
            if ast.unparse(node) not in imports:
                imports.append(ast.unparse(node))
        
        def visit_ImportFrom(self, node):
            if ast.unparse(node) not in imports:
                imports.append(ast.unparse(node))

    ImportCollector().visit(tree)
    return "\n".join(imports)

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

def set_timeout(seconds):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

def main(file1, file2, test_dir):
    try:
        set_timeout(10)
        content1 = clean_code(file1)
        content2 = clean_code(file2)
        inputs1 = get_inputs(file1, test_dir)
        inputs2 = get_inputs(file2, test_dir)
        if len(inputs1) == 0 or len(inputs2) == 0:
            return [], ""
        test_input = f"{inputs1[0]}\n{inputs2[0]}"
        input_nodes1 = extract_input_nodes(content1)
        input_nodes2 = extract_input_nodes(content2)
        import_nodes1 = extract_import_nodes(content1)
        import_nodes2 = extract_import_nodes(content2)
        beginning_code = "\n".join([import_nodes1, import_nodes2, input_nodes1, input_nodes2])
        code_content1 = remove_lines_from_code(content1, input_nodes1)
        code_content1 = remove_lines_from_code(code_content1, import_nodes1)
        code_content2 = remove_lines_from_code(content2, input_nodes2)
        code_content2 = remove_lines_from_code(code_content2, import_nodes2)
        compilable_permutations1 = test_permutations_without_input(code_content1, beginning_code)
        compilable_permutations2 = test_permutations_without_input(code_content2, beginning_code)
        combined_compilable = combine_and_test(compilable_permutations1, compilable_permutations2, test_input, beginning_code)
        signal.alarm(0)
        return combined_compilable, test_input
    except:
        return [], ""
    
def get_inputs(file1, test_dir):
    file_id = file1.split("/")[-1].split(".py")[0]
    inputs = []
    for dir, _, files in os.walk(test_dir):
        for file in files:
            if "in" in file and file_id in file:
                file_path = os.path.join(dir, file)
                input_content = utils.read_file(file_path)
                inputs.append(input_content)
    return inputs
    
if __name__ == "__main__":
    # args = sys.argv[1:]
    # file1 = args[0]
    # file2 = args[1]
      

    file1 = ".tmp_patches/s681105182/s681105182.py-c6c0e5a15fbb3f386051120bddbb4c4c63cecb45d477852c37901a34ad4ac6ee"
    file2 = ".tmp_patches/s681105182/s681105182.py-14d128ef48f9fcb420849d28882fa6f522bff694573140987a2cba0d64650754"
    test_dir = "/home/yang/PLTranslationEmpirical/dataset/codenet/Python/TestCases"
    combined_compilable, test_input = main(file1, file2, test_dir)
    for line in combined_compilable:
        print(line)
    
    #  /home/yang/PLTranslationEmpirical/dataset/avatar/Python/TestCases


