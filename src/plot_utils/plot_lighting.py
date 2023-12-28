import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

results_file = 'track_images/lighting_inference/results.csv'
path = '/'.join(results_file.split('/')[:-1])
df = pd.read_csv(results_file)
df = df.fillna(0)

metric_names = [x for x in df.columns if x != 'sequence']
intervals_names = {x.replace('/', ''): x for x in ['1s', '2s', '5s', '10s', '20s']}
intervals = [x for x in intervals_names.keys()]

intervals_color = cm.rainbow(np.linspace(0, 1, len(intervals)))
intervals_colormap = {interval: c for interval, c in zip(intervals, intervals_color)}

metric_dict = {x: [] for x in intervals}

for interval in intervals:
    for met in metric_names:
        stat = (df.loc[df['sequence']==interval][met]).values[0]
        metric_dict[interval].append(stat)

# Bar graph
x = np.arange(len(metric_names))
width = 0.1
multiplier = 0

fig, ax = plt.subplots(figsize=(10,6))

for interval, stats in metric_dict.items():
    offset = width*multiplier - 0.25
    rects = ax.bar(x + offset, stats, width, label=intervals_names[interval], color=intervals_colormap[interval])
    # ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('Performance')
ax.set_xlabel('Metric')
ax.set_title('Trackformer Performance over Lighting Interval Length')
ax.set_xticks(x+width)
ax.set_xticklabels(metric_names)
ax.set_ylim(-0.3, 1)
ax.legend(bbox_to_anchor=(1.05, 1.05))
ax.grid(True, which='major', axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{path}/bargraph_lighting.svg')


# Line graph
fig, ax = plt.subplots(figsize=(10,6))

x = [int(x[:-1]) for x in intervals]

met_color = cm.rainbow(np.linspace(0, 1, len(metric_names)))
met_colormap = {met: c for met, c in zip(metric_names, met_color)}

for metric in metric_names:
    rows = df.loc[df['sequence'].isin(intervals)]
    y = []
    for idx, row in rows.iterrows():
        y.append(row[metric])
    ax.plot(x, y, linewidth=2.0, color=met_colormap[metric], label=metric)

ax.set_ylabel('Performance')
ax.set_xlabel('Length of Interval (s)')
ax.set_title('Trackformer Performance over Lighting Interval Length')
ax.set_ylim(-0.3, 1)
ax.legend(bbox_to_anchor=(1.05, 1.05))
ax.grid(True, which='major', axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{path}/linegraph_lighting.svg')