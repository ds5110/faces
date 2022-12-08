#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:38:28 2022

@author: jhautala
"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# internal
from util import meta_cache
from util.model import nose_i, outer_canthi
from util.pre import rotate
from util.plot import plot_image
from util.column_names import x_cols_cenrot,\
    x_cols_norm_cenrot,\
    y_cols_norm_cenrot


save_fig = False

# for col in df.columns:
#     print(f'{col}: {df[col].dtype}')

df = meta_cache.get_meta()
row_id = 690
anno = meta_cache.get_image(row_id)
rotated = rotate(anno)
x_coords = rotated.get_x()
x_min = x_coords.min()
x_max = x_coords.max()
x_mid = (x_min + x_max)/2
x_nose = x_coords[nose_i]
y_nose = rotated.get_y()[nose_i]
# row = df.iloc[row_id]
# x_cenrot = row[x_cols_norm_cenrot].to_numpy()

fig, ax = plt.subplots(figsize=(10, 10))
plot_image(
    rotated,
    annotate='scatter',
    ax=ax,
)
canthi = rotated.get_coords()[outer_canthi]
ax.plot(canthi[:,0], canthi[:,1], marker='|')
plt.tight_layout()
if save_fig:
    plt.savefig(
        'figs/yaw_analysis.png',
        dpi=300,
        bbox_inches='tight'
    )
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
plot_image(
    rotated,
    annotate='scatter',
    ax=ax,
)
ax.title.set_text('roll analysis')
ax.axvline(x_min, c='tab:gray', linestyle='--')
ax.axvline(x_max, c='tab:gray', linestyle='--')
ax.plot([x_min, x_max], [y_nose, y_nose])
ax.plot([x_nose, x_mid], [y_nose, y_nose], c='tab:red', marker='|')

plt.tight_layout()
if save_fig:
    plt.savefig(
        'figs/roll_analysis.png',
        dpi=300,
        bbox_inches='tight'
    )
plt.show()