#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:10:06 2022

@author: jhautala
"""

from functools import reduce
import numpy as np

# intra-project
from util.local_cache import cache
from util.plot import plot_image, plot_coords
from util.pre import rotate
from util.model import cat_cols, cenrot_cols

# load the labels data
df = cache.get_meta('decorated')

def annotated_plot(types=None):
    if types is None:
        types = [None,'scatter','scatternum','spline','splinelabel']
    for i in range(10):
        anno = cache.get_image(i)
        for annotate in types:
            plot_image(
                anno,
                annotate=annotate,
                cross=False,
                save_fig=False,
            )
            plot_image(
                rotate(anno),
                annotate=annotate,
                cross=True,
                save_fig=False,
            )

def test_challenging(save_fig=False):
    # [f'{i:b}'.rjust(4,'0') for i in range(1,16)]
    masks = [df[col] == 1 for i, col in enumerate(cat_cols)]
    for combo in [f'{i:b}'.rjust(4,'0') for i in range(1,16)]:
        desc = ', '.join(col for i, col in enumerate(cat_cols) if combo[i] == '1')
        mask = [m if combo[i] == '1' else ~m for i, m in enumerate(masks)]
        tmp = df[reduce(lambda a,b: a & b, mask)]
        if tmp.shape[0] == 0:
            print(f'no rows found for combination! {desc}')
            continue;
        print(f'found {tmp.shape[0]} rows ({desc})')
        
        row_id = tmp.index[0]
        anno = cache.get_image(row_id,desc)
        plot_image(
            rotate(anno),
            annotate='spline',
            cross=True,
            grayscale=True,
            save_fig=save_fig,
        )
        
        # plot geometry without image, to verify coords
        plot_image(
            rotate(anno),
            annotate='spline',
            cross=True,
            grayscale=True,
            skip_img=True,
            save_fig=save_fig,
        )
        coords = np.stack(
            [tmp[cols].loc[row_id,:].values for cols in cenrot_cols],
            1
        )
        plot_coords(
            coords,
            tmp.loc[row_id,:]['width'],
            tmp.loc[row_id,:]['height'],
            save_fig=save_fig,
        )

if __name__ == '__main__':
    # annotated_plot()
    # annotated_plot(['spline'])
    test_challenging()
