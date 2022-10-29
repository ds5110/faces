#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:10:06 2022

@author: jhautala
"""

import matplotlib.pyplot as plt

# intra-project
from util.local_cache import cache
from util.plot import plot_image

# load the labels data
df = cache.get_meta()

def scrape_all():
    for i in range(df.shape[0]):
        cache.get_image(i)

def simple_plot():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cache.get_image2(1).im_fun())
    plt.tight_layout()
    plt.show()

def annotated_plot():
    types = [None,'scatter','scatternum','spline','splinelabel']
    for i in range(10):
    
        series = df.iloc[i,:]
        img = cache.get_image(
            path=series['image-set'],
            file=series['filename'],
        )
        for annotate in types:
            plot_image(
                img,
                df,
                i,
                annotate=annotate,
                save_fig=False
            )
    
if __name__ == '__main__':
    # scrape_all()
    simple_plot()
    # annotated_plot()
