#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:10:06 2022

@author: jhautala
"""

# intra-project
from util.local_cache import cache
from util.plot import plot_image
from util.pre import rotate

# load the labels data
df = cache.get_meta()

def scrape_all():
    for i in range(df.shape[0]):
        cache.get_image(i)

def annotated_plot():
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
    
if __name__ == '__main__':
    # scrape_all()
    annotated_plot()
