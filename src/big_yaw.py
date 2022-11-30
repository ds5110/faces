#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:51:47 2022

@author: jhautala
"""

from util import meta_cache
from util.plot import plot_image
from util.pre import to_deg

baby_df = meta_cache.get_meta('baby')
save_fig = False

tilted = baby_df['tilted'] == 1
turned = baby_df['turned'] == 1
unlabeled  = baby_df[~tilted & ~turned]\
    .sort_values(by='yaw_abs', ascending=False)
for i in range(10):
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
