import os
import sys
import csv
import ast
import shutil

ids = ["s913338871","s615966179","s454986054","s870744379","s706119740","s453478465","s103354654","s500828884","s858647829","s214672212","s197514717","s831456551","s695100304","s131946120","s316832655","s251858505","s028530838","s031917494","s592421258","s087080706","s485638370","s142939923","s404600540","s013976107","s025428739","s000375264","s934308496","s091400419","s191853417","s052231578"]

# codenet_results_24h.csv
with open("humaneval_results.csv", mode ='r')as file:
  csvFile = csv.reader(file)
  for line in csvFile:
        if "source_file" in line:
            continue
        if "crossover" in str(line):
            continue
        files = line[15]
        max_file = line[16]
        if max_file != "":
            for f in ast.literal_eval(files):
                if ".tmp_patches/" in f:
                    shutil.copy(f, "humaneval_transformations")
        else:
            for key in ast.literal_eval(files):
                if ".tmp_patches/" in key:
                    shutil.copy(key, "humaneval_transformations")
                break
        # exit(0)