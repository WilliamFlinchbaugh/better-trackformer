import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl
import csv

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

path = 'track_images/fpn_compare/'
metric_names = ['IDF1', 'MOTA']
met_colormap = {'MOTA': 'red', 'IDF1': 'blue'}

distances = ['25m', '50m', '100m', '150m', '200m', '300m', '400m', '500m', '750m', '1000m']

# --------- DISTANCE ---------
old_file = 'track_images/distance_inference/results.csv'
old_path = '/'.join(old_file.split('/')[:-1])
old_df = pd.read_csv(old_file)
old_df = old_df.fillna(0)

fpn_file = 'track_images/fpn_dist/fpn_dist_results.csv'
fpn_path = '/'.join(fpn_file.split('/')[:-1])
fpn_df = pd.read_csv(fpn_file)
fpn_df = fpn_df.fillna(0)

fig, ax = plt.subplots(figsize=(10,6))
ax.grid(True, which='major', axis='both', alpha=0.3)
ax.axhline(y=0, color='k', linestyle='--')

x = [int(x[:-1]) for x in distances]

# plot untuned
for metric in metric_names:
    rows = old_df.loc[old_df['sequence'].isin(distances)]
    y = []
    for idx, row in rows.iterrows():
        y.append(row[metric])
    ax.plot(x, y, linewidth=2.0, color=met_colormap[metric], label=f'Untuned {metric}', linestyle='solid')
    
# plot tuned
for metric in metric_names:
    rows = fpn_df.loc[fpn_df['sequence'].isin(distances)]
    y = []
    for idx, row in rows.iterrows():
        y.append(row[metric])
    ax.plot(x, y, linewidth=2.0, color=met_colormap[metric], label=f'Tuned {metric}', linestyle='dashdot')

plt.xscale('log')
ax.set_xticks(x)
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
ax.set_ylabel('Performance')
ax.set_xlabel('Distance (m)')
ax.set_title('Trackformer Performance over Distance')
ax.set_ylim(-1, 1)
ax.set_xlim(x[0], x[-1])
ax.legend(bbox_to_anchor=(1.05, 1.05))
plt.tight_layout()
plt.savefig(f'{path}distance_plot.svg')

# -------- BBOX SIZE --------
old_file = 'track_images/distance_inference/results.csv'
old_path = '/'.join(old_file.split('/')[:-1])
old_df = pd.read_csv(old_file)
old_df = old_df.fillna(0)

fpn_file = 'track_images/fpn_dist/fpn_dist_results.csv'
fpn_path = '/'.join(fpn_file.split('/')[:-1])
fpn_df = pd.read_csv(fpn_file)
fpn_df = fpn_df.fillna(0)

gt_root = 'data/distances'

# Average the avg BB area over each sequence
dist_avg_areas = {dist: 0 for dist in distances}
n_seqs = 0
for distance in distances:
    dist_root = os.path.join(gt_root, distance)
    seq_areas = []
    for seq in os.listdir(dist_root):
        n_seqs += 1
        n_boxes = 0
        areas = []
        
        gt_file_path = os.path.join(dist_root, seq, 'gt', 'gt.txt')
        with open(gt_file_path, "r") as gt_file:
            reader = csv.reader(gt_file, delimiter=',')
            for row in reader:
                n_boxes += 1
                bbox = [float(row[2]), float(row[3]), float(row[4]), float(row[5])]
                areas.append(bbox[2] * bbox[3]) # add individual BB areas
                
        seq_areas.append(sum(areas) / n_boxes) # average all areas in the sequence
    dist_avg_areas[distance] = sum(seq_areas) / n_seqs # average all avg sequence areas of a certain distance

fig, ax = plt.subplots(figsize=(10,6))
ax.axhline(y=0, color='k', linestyle='--')
ax.grid(True, which='major', axis='both', alpha=0.3)

x = [dist_avg_areas[dist] for dist in distances]

# plot untuned
for metric in metric_names:
    rows = old_df.loc[old_df['sequence'].isin(distances)]
    y = []
    for idx, row in rows.iterrows():
        y.append(row[metric])
    ax.plot(x, y, linewidth=2.0, color=met_colormap[metric], label=f'W/O FPN {metric}', linestyle='solid')
    
# plot tuned
for metric in metric_names:
    rows = fpn_df.loc[fpn_df['sequence'].isin(distances)]
    y = []
    for idx, row in rows.iterrows():
        y.append(row[metric])
    ax.plot(x, y, linewidth=2.0, color=met_colormap[metric], label=f'W/ FPN {metric}', linestyle='dashdot')

plt.xscale('log')
ax.set_xticks(x)
ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.get_xaxis().set_tick_params(which='minor', size=0)
ax.get_xaxis().set_tick_params(which='minor', width=0) 
ax.set_ylabel('Performance')
ax.set_xlabel('Bounding Box Area (px)')
ax.set_title('Trackformer Performance over Bounding Box Area')
ax.set_ylim(-1, 1)
ax.set_xlim(x[0], x[-1])
ax.legend(bbox_to_anchor=(1.05, 1.05))
plt.tight_layout()
plt.savefig(f'{path}bbox_plot.svg')
