#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:51:47 2022

@author: jhautala
"""

from util import meta_cache
from util.plot import plot_image
from util.pre import to_deg
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_hist(
        df,
        desc,
        x='yaw_abs',
        y=None,
        file_suff=None,
        hue=None,
        save_fig=False,
):
    tmp = df.copy()
    tmp[f'{x}_deg'] = tmp[x] * to_deg
    if y:
        tmp[f'{y}_deg'] = tmp[y] * to_deg
    ax = sns.histplot(
        data=tmp,
        x=f'{x}_deg',
        y=f'{y}_deg' if y else None,
        hue=hue,
        multiple='stack',
        # stat='density',
    )
    ax.set_title(desc)
    plt.tight_layout()
    if save_fig:
        filename = 'angle_hist'
        if file_suff:
            filename += f'_{file_suff}'
        plt.savefig(
            f'figs/{filename}.png',
            dpi=300,
            bbox_inches='tight',
        )
    plt.show()


# load data and config
baby_df = meta_cache.get_meta('baby')
save_fig = False
top_ten = True


# create masks
tilted = baby_df['tilted'] == 1
turned = baby_df['turned'] == 1
only_tilted = tilted & ~turned
only_turned = ~tilted & turned
neither = (~tilted & ~turned)
both = tilted & turned

# create df of 'missed' labels for yaw (expect tilted)
unlabeled  = baby_df[neither]\
    .sort_values(by='yaw_abs', ascending=False)

n = unlabeled.shape[0]
if top_ten:
    n = min(10, n)
for i in range(n):
    row = unlabeled.iloc[i]
    row_id = unlabeled.index[i]
    anno = meta_cache.get_image(row_id, baby=True)
    anno.desc = ['big yaw']
    plot_image(
        anno,
        save_fig=save_fig,
        subtitle=[
            f'{to_deg * row["yaw_abs"]:.2f} degrees estimated absolute yaw',
            'not labeled "tilted" nor "turned"',
        ],
    )

# # simple histograms
# plot_hist(
#     baby_df,
#     'All images')
# plot_hist(
#     unlabeled,
#     'Neither "tilted" nor "turned"',
# )

# plot_hist(
#     baby_df[tilted],
#     'Tilted',
#     # x='roll_abs',
#     file_suff='tilted',
#     # hue='label',
#     save_fig=save_fig,
# )
# plot_hist(
#     baby_df[~tilted],
#     'Not tilted',
#     # x='roll_abs',
#     file_suff='tilted',
#     # hue='label',
#     save_fig=save_fig,
# )

# dfs = []
# for (mask, desc) in [
#         (tilted, 'tilted'),
#         (~tilted, 'not tilted'),
# ]:
#     tmp = baby_df[mask] if mask is not None else baby_df.copy()
#     tmp['label'] = desc
#     dfs.append(tmp)
# tmp = pd.concat(dfs)
# # tmp = tmp[tmp['yaw_abs'] > .35]
# # tmp = tmp[tmp['yaw_abs'] < 1.745]
# # tmp = tmp[tmp['yaw_abs'] > .3]
# # tmp = tmp[tmp['yaw_abs'] < 1.8]
# tmp = tmp[tmp['yaw_abs'] > np.pi/4]
# tmp = tmp[tmp['yaw_abs'] < np.pi/2]

# plot_hist(
#     tmp,
#     'All images per "tilted"',
#     file_suff='stacked',
#     hue='label',
#     save_fig=save_fig,
# )


# # create label column
# dfs = []
# for (mask, desc) in [
#         (only_tilted, 'tilted'),
#         (both, 'both'),
#         (only_turned, 'turned'),
#         (neither, 'neither'),
# ]:
#     tmp = baby_df[mask]
#     tmp['label'] = desc
#     dfs.append(tmp)
# tmp = pd.concat(dfs)


# plot_hist(
#     tmp,
#     'All images per angle labels',
#     # x='roll_abs',
#     file_suff='stacked',
#     hue='label',
#     save_fig=save_fig,
# )

