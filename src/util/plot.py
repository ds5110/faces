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

from util.model import landmark68, x_cols, y_cols

def plot_image(
        img,
        df,
        row_id=None,
        series=None,
        annotate=None,
        save_fig=False
):
    '''
    Either df/row_id or series is required, to provide landmark points.
    Parameters
    ----------
    img : Pillow Image
        The image to render.
    df : Pandas DataFrame
        The data frame.
    row_id : int, optional
        The row index (per iloc). The default is None.
    series : Pandas Series, optional
        The row. The default is None.
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
    if not series:
        series = df.iloc[row_id,:]
    category = series['image-set']
    filename = series['filename']
    title = f'{category}/{filename}'
    if row_id is not None:
        title += f' (row {row_id})'
    
    # get the image data
    # img.convert('YCbCr')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    if annotate:
        if annotate.startswith('spline'):
            for f in landmark68.features:
                xx = series[[f'gt-x{i}' for i in f.idx]].astype(float).values
                yy = series[[f'gt-y{i}' for i in f.idx]].astype(float).values
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
                splines = [UnivariateSpline(distance, point, k=2, s=.1) for point in points]
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
                series[x_cols],
                series[y_cols],
                s=6,
                linewidth=.5,
                c='lime',
                edgecolors='black',
            )
            if 'scatternum' == annotate:
                for i, x_col in enumerate(x_cols):
                    ax.annotate(
                        f'{i}',
                        (series[x_col], series[y_cols[i]]),
                        fontsize=6,
                    )
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        path = f'./figs/temp/images/{category}'
        if not Path(path).exists():
            os.makedirs(path)
        # NOTE: File names would look nicer if we remove the original
        #       suffix here, but it will be easier to tie back to the
        #       original image if we don't.
        if annotate:
            filename += f'_{annotate}'
        plt.savefig(
            f'{path}/{filename}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()
