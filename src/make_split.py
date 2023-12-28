import shutil
import os
import random
import json
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument('set_folder')
argParser.add_argument('split_name')
argParser.add_argument('-v', '--variable', help='if the dataset contains a variable (distance, movement, lighting, etc.)', action='store_true')
args = argParser.parse_args()

data_folder = 'data'
set_folder = os.path.join(data_folder, args.set_folder)
seq_folder = os.path.join(set_folder, 'seqs')
if args.variable:
    variants = {x.split('_')[-1] for x in os.listdir(seq_folder)}
    all_seqs = {x: [] for x in variants}
    for x in os.listdir(seq_folder):
        all_seqs[x.split('_')[-1]].append(x)
else:
    all_seqs = os.listdir(seq_folder)

split_folder = os.path.join(set_folder, args.split_name)

if os.path.exists(split_folder):
    shutil.rmtree(split_folder)

test_folder = os.path.join(split_folder, 'test')
train_folder = os.path.join(split_folder, 'train')

seq_split = {
    'test': [],
    'train': []
}

if args.variable:
    for var, seqs in all_seqs.items():
        n_test_sets = int(len(seqs) * 0.3)
        n_train_sets = len(seqs) - n_test_sets

        # Choose the split
        test_seqs = random.sample(seqs, n_test_sets)
        train_seqs = list(set(seqs).difference(set(test_seqs)))

        seq_split['test'].extend(test_seqs)
        seq_split['train'].extend(train_seqs)
            
else:
    n_test_sets = int(len(all_seqs) * 0.3)
    n_train_sets = len(all_seqs) - n_test_sets

    # Choose the split
    test_seqs = random.sample(all_seqs, n_test_sets)
    train_seqs = list(set(all_seqs).difference(set(test_seqs)))

    seq_split['test'].extend(test_seqs)
    seq_split['train'].extend(train_seqs)
    
# Write json for the split lists
with open(os.path.join(set_folder, f'{args.split_name}.json'), 'w') as f:
    json.dump(seq_split, f, indent=4)
