#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 20:31:24 2022

@author: jhautala
"""

from util import meta_cache
from util.plot import plot_image
from util.pre import rotate

df = meta_cache.get_meta()
baby_df = meta_cache.get_meta('baby')
save_fig = False

anno_img = meta_cache.get_image(4, baby=True)
for annotate in ['scatternum', 'splines']:
    plot_image(
        anno_img,
        annotate=annotate,
        save_fig=save_fig,
    )

for (row_id, annotate) in [
        (4, None),
        (1, 'scatter'),
        (0, 'splinelabel')
]:
    plot_image(
        rotate(meta_cache.get_image(row_id, baby=True)),
        annotate=annotate,
        cross=True,
    )
