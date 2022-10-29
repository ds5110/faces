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
    anno = cache.get_image2(1)
    rot = rotate(anno)
    rot_img = rot.get_image()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rot_img)
    if scatter:
        ax.scatter(
            rot.coords[0],
            rot.coords[1],
            s=6,
            linewidth=.5,
            c='lime',
            edgecolors='black',
        )
    if cross:
        center= np.array([rot_img.width/2, rot_img.height/2])
        ax.axhline(y=center[1])
        ax.axvline(x=center[0])
    plt.title('test')
    plt.tight_layout()
    plt.show()

def annotated_plot():
    types = [None,'scatter','scatternum','spline','splinelabel']
    for i in [0]:#range(10):
    
        anno = cache.get_image2(i)
        for annotate in types:
            plot_image(
                anno,
                annotate=annotate,
                cross=True,
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
    # simple_plot()
    annotated_plot()
