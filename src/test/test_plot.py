#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:10:06 2022

@author: jhautala
"""

import matplotlib.pyplot as plt
import numpy as np

# intra-project
from util.local_cache import cache
from util.plot import plot_image
from util.pre import rotate

# load the labels data
df = cache.get_meta()

def scrape_all():
    for i in range(df.shape[0]):
        cache.get_image(i)

def simple_plot(scatter=False,cross=False):
    anno = cache.get_image(1)
    for do_rot in [False, True]:
        if do_rot:
            anno = rotate(anno)
        img = anno.get_image()
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        if scatter:
            ax.scatter(
                anno.get_x(),
                anno.get_y(),
                s=6,
                linewidth=.5,
                c='lime',
                edgecolors='black',
            )
        if cross:
            center= np.array([img.width/2, img.height/2])
            ax.axhline(y=center[1])
            ax.axvline(x=center[0])
        plt.title('test')
        plt.tight_layout()
        plt.show()

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
                rotate(anno), # rotated images have a distinct 'desc'
                annotate=annotate,
                cross=True,
                save_fig=False,
            )
    
if __name__ == '__main__':
    # scrape_all()
    # simple_plot(scatter=True,cross=True)
    annotated_plot()
