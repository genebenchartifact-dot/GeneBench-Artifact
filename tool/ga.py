import ast
import astor
import astunparse
import os
import random
import utils

# def get_all_applicable_rules(python_code):
    

def transformation_ga(source_file, applicable_rules):
    original_code = utils.read_file(source_file)
    
    return update_content, applicable_rules