#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 17:02:56 2022

@author: jhautala
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

from util.model import landmark68

def plot_image(
        anno,
        annotate=None,
        cross=False,
        save_fig=False,
):
    '''
    Either df/row_id or series is required, to provide landmark points.
    Parameters
    ----------
    anno : AnnoImg
        The image to render.
    annotate : str, optional
        The type of annotations to draw on the image:
            - 'scatter': the landmark points
            - 'scaternum': landmark points with indices
            - 'spline': best-fit splines between landmark points
            - 'splinelabel': best-fit splines with feature names
        The default is None.
    save_fig : bool, optional
        Pass True to save result to 'figs' directory. The default is False.

    Returns
    -------
    None.

    '''
    image_set = anno.image_set
    filename = anno.filename
    desc = f' (row {anno.row_id})' if anno.row_id is not None else ''
    if anno.desc:
        desc += f' {anno.desc}'
    title = f'{image_set}/{filename}' + desc
    X = anno.get_x()
    Y = anno.get_y()
    img = anno.get_image()
    
    # get the image data
    # img.convert('YCbCr')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    if cross:
        center= np.array([img.width/2, img.height/2])
        ax.axhline(y=center[1])
        ax.axvline(x=center[0])
    
    if annotate:
        if annotate.startswith('spline'):
            for f in landmark68.features:
                xx = anno.get_x()[f.idx]
                yy = anno.get_y()[f.idx]
                if pd.isna(xx).any(): continue;
                if pd.isna(yy).any(): continue;
                
                points = np.stack((xx,yy))
                distance = np.cumsum(
                    np.sqrt(np.sum(
                        np.diff(points.T, axis=0)**2,
                        axis=1
                    ))
                )
                if not distance[-1]: continue;
                
                distance = np.insert(distance, 0, 0)/distance[-1]
                splines = [UnivariateSpline(distance, point, k=2, s=.2) for point in points]
                points_fitted = np.vstack(
                    [spline(np.linspace(0, 1, 64)) for spline in splines]
                )
                plt.plot(
                    *points_fitted,
                    linestyle='-',
                    linewidth='1',
                    c='fuchsia',
                )
                
                if 'splinelabel' == annotate:
                    # TODO: come up with a way to avoid overlapping labels
                    mid_x = xx[len(xx)//2]
                    if len(xx) % 2 == 0:
                        mid_x = (mid_x + xx[len(xx)//2 - 1])/2
                    mid_y = yy[len(yy)//2]
                    if len(yy) % 2 == 0:
                        mid_y = (mid_y + yy[len(yy)//2 - 1])/2
                    ax.annotate(
                        f'{f.desc}',
                        (mid_x, mid_y),
                        fontsize=6,
                    )
                else:
                    plt.plot(
                        *points,
                        'o',
                        markersize=1,
                        c='white',
                    )
        if annotate and annotate.startswith('scatter'):
            ax.scatter(
                X,
                Y,
                s=6,
                linewidth=.5,
                c='lime',
                edgecolors='black',
            )
            if 'scatternum' == annotate:
                for i in range(len(X)):
                    ax.annotate(
                        f'{i}',
                        (X[i], Y[i]),
                        fontsize=6,
                    )
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        path = f'./figs/images/{image_set}'
        if not Path(path).exists():
            os.makedirs(path)
        # NOTE: File names would look nicer if we remove the original
        #       suffix here, but it will be easier to tie back to the
        #       original image if we don't.
        if annotate:
            filename += f'_{annotate}'
        if anno.desc:
            filename += f'_{anno.desc}'
        plt.savefig(
            f'{path}/{filename}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()
