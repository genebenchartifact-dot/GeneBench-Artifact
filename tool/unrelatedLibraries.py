import os
import sys
import ast
import utils

unrelated_libs = """
import os
import itertools
import bisect
import heapq
import pprint
import fractions
from io import BytesIO, IOBase
from itertools import permutations, combinations, product
from string import ascii_lowercase, ascii_uppercase, digits
"""

def add_unrelated_libs(import_list):
    unparsed_imporst = []
    toadd_import_list = []
    for node in import_list:
        if ast.dump(node) not in unparsed_imporst:
            unparsed_imporst.append(ast.dump(node))
    root = ast.parse(unrelated_libs)
    import_nodes = utils.get_imports(root)
    if len(import_nodes) != len(import_list):
        print("Rule add_unrelated_libs is applicable!")
    for node in import_nodes:
        if ast.dump(node) not in unparsed_imporst:
            toadd_import_list.append(node)
            print("Will add " + ast.unparse(node))
    return toadd_import_list