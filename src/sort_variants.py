import argparse
import configparser
import csv
import json
import os
import shutil

argParser = argparse.ArgumentParser()
argParser.add_argument('set_folder')
args = argParser.parse_args()

set_folder = os.path.join('data', args.set_folder)
seq_folder = os.path.join(set_folder, 'seqs')

variants = {x.split('_')[-1] for x in os.listdir(seq_folder)}
all_seqs = {x: [] for x in variants}
for x in os.listdir(seq_folder):
    all_seqs[x.split('_')[-1]].append(x)
    
for x in variants:
    os.mkdir(os.path.join(set_folder, x))
    for seq in all_seqs[x]:
        shutil.copytree(os.path.join(seq_folder, seq), os.path.join(set_folder, x, seq))
    

