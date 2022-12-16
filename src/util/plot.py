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
from PIL import ImageOps

from util.model import landmark68


class Case:
    def __init__(self, name, color, mask):
        self.name = name
        self.color = color
        self.mask = mask


def fix_axes(X,Y,ax,flip_y=True):
    data = np.stack([X,Y],axis=1)
    mins = np.amin(data,axis=0)
    maxs = np.amax(data,axis=0)
    extents = maxs - mins
    max_extent = np.max(extents)
    for i, (lim, fun) in enumerate([
        (ax.get_xlim(), ax.set_xlim),
        (ax.get_ylim(), ax.set_ylim)
    ]):
        buff = (max_extent - extents[i])/2
        fun((lim[0]-buff,lim[1]+buff))
    if flip_y:
        ax.set_ylim(ax.get_ylim()[::-1])
    
def plot_image(
        anno,
        subtitle=None,
        scatter_points=None,
        ref_point=None,
        annotate=None,
        cross=False,
        grayscale=False,
        skip_img=False,
        unit_scale=False,
        save_fig=False,
        ax=None,
):
    '''
    Either df/row_id or series is required, to provide landmark points.
    Parameters
    ----------
    anno : AnnoImg
        The image to render.
    subtitle : list of str, optional
        Additional lines of text below the title. The default is None.
    scatter_points : list of int
        Landmark indices to print (excldues unlisted landmarks).
        The default is None.
    ref_point : list of int ['x', 'y']
        Add a swhite circle centered at given 'x' and 'y'.
        The default is None.
    annotate : str, optional
        The type of annotations to draw on the image:
            - 'scatter': the landmark points
            - 'scaternum': landmark points with indices
            - 'spline': best-fit splines between landmark points
            - 'splinelabel': best-fit splines with feature names
            - 'lines': similar to splines, but without interpolation
        The default is None.
    cross : bool
        True to include blue crosshairs at image center.
        The default is False.
    grayscale : bool or str
        If True then convert image to grayscale (using default colormap 'viridis').
        If str then interpret as the name of a colormap for grayscale.
        The default is False.
    skip_img : bool
        True to skip raw image data. The default is False.
    unit_scale : bool
        True to apply hacks for normalized landmarks. The default is False.
    save_fig : bool, optional
        True to save result to 'figs' directory. The default is False.
    ax: pyplot Axes
        The Axes to plot onto. When provided, we skip showing/saving
        the plot (caller owns). The default is None.

    Returns
    -------
    None.

    '''
    img = anno.get_image()
    image_set = anno.image_set
    filename = anno.filename
    desc = f' (row {anno.row_id})' if anno.row_id is not None else ''
    if anno.desc:
        desc += f' - {", ".join(anno.desc)}'
    if subtitle:
        desc = ' \n '.join([desc, *subtitle])
    title = f'{image_set}/{filename}' + desc
    X = anno.get_x()
    Y = anno.get_y()
    if grayscale:
        if isinstance(grayscale, str):
            cmap = grayscale
        else:
            cmap = 'viridis'
        # NOTE: Extracting luma component produces the same result
        #       as ImageOps.grayscale.
        # y,u,v = img.convert('YCbCr').split()
        img = ImageOps.grayscale(img)
    else:
        cmap = None
    
    # get the image data
    # img.convert('YCbCr')
    
    if ax is None:
        own_axes = True
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        own_axes = False
    if img is not None and not skip_img:
        ax.imshow(img, cmap=cmap)
        if cross:
            center = np.array([img.width/2, img.height/2])
            ax.axhline(y=center[1])
            ax.axvline(x=center[0])
    
    if annotate:
        if annotate.startswith('spline'):
            s = .000001 if unit_scale else .2
            for f in landmark68.features:
                xx = X[f.idx]
                yy = Y[f.idx]
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
                splines = [UnivariateSpline(distance, point, k=2, s=s) for point in points]
                points_fitted = np.vstack(
                    [spline(np.linspace(0, 1, 64)) for spline in splines]
                )
                ax.plot(
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
                    ax.plot(
                        *points,
                        'o',
                        markersize=1,
                        c='tab:blue' if skip_img else 'white',
                    )
        elif annotate == 'lines':
            for f in landmark68.features:
                xx = X[f.idx]
                yy = Y[f.idx]
                if pd.isna(xx).any(): continue;
                if pd.isna(yy).any(): continue;

                points = np.stack((xx, yy))
                ax.plot(
                    *points,
                    linestyle='-',
                    linewidth='1',
                    c='fuchsia',
                )
                ax.plot(
                    *points,
                    'o',
                    markersize=1,
                    c='tab:blue' if skip_img else 'white',
                )
        if annotate and annotate.startswith('scatter'):
            if scatter_points is None:
                scatter_points = range(68)
            ax.scatter(
                X[scatter_points],
                Y[scatter_points],
                s=6,
                linewidth=.5,
                c='lime',
                edgecolors='black',
            )
            if 'scatternum' == annotate:
                for i in scatter_points:
                    ax.annotate(
                        f'{i}',
                        (X[i], Y[i]),
                        fontsize=6,
                    )
    if ref_point is not None:
        ax.scatter(
            [ref_point[0]],
            [ref_point[1]],
            s=10,
            c='white'
        )
    if skip_img:
        fix_axes(X,Y,ax)
    if unit_scale:
        ax.set_xlim(-1, 1)
        ax.set_ylim(1, -1)
    
    # skip display for unowned axes
    # NOTE: this is a quick fix to support multiplot
    # TODO: revisit...
    if not own_axes:
        return
    
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
            filename += f'_{"_".join(anno.desc)}'
        plt.savefig(
            f'{path}/{filename}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()

def plot_coords(
        coords,
        width,
        height,
        save_fig=False,
):
    image_set = 'test'
    filename = 'cenrot'
    title = f'{image_set}/{filename}'
    X = coords[:,0]
    Y = coords[:,1]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    center= np.array([width/2, height/2])
    ax.axhline(y=center[1])
    ax.axvline(x=center[0])
    
    # for f in landmark68.features:
    #     xx = X[f.idx]
    #     yy = X[f.idx]
    #     if pd.isna(xx).any(): continue;
    #     if pd.isna(yy).any(): continue;
        
        #------ calculate splines
        # points = np.stack((xx,yy))
        # distance = np.cumsum(
        #     np.sqrt(np.sum(
        #         np.diff(points.T, axis=0)**2,
        #         axis=1
        #     ))
        # )
        # if not distance[-1]: continue;
        
        # distance = np.insert(distance, 0, 0)/distance[-1]
        # splines = [UnivariateSpline(distance, point, k=2, s=.2) for point in points]
        # points_fitted = np.vstack(
        #     [spline(np.linspace(0, 1, 64)) for spline in splines]
        # )
        
        # # plot splines
        # plt.plot(
        #     *points_fitted,
        #     linestyle='-',
        #     linewidth='1',
        #     c='fuchsia',
        # )
        
        # # plot spline labels
        # # TODO: come up with a way to avoid overlapping labels
        # mid_x = xx[len(xx)//2]
        # if len(xx) % 2 == 0:
        #     mid_x = (mid_x + xx[len(xx)//2 - 1])/2
        # mid_y = yy[len(yy)//2]
        # if len(yy) % 2 == 0:
        #     mid_y = (mid_y + yy[len(yy)//2 - 1])/2
        # ax.annotate(
        #     f'{f.desc}',
        #     (mid_x, mid_y),
        #     fontsize=6,
        # )
        
        # # plot landmarks as white dots
        # plt.plot(
        #     *points,
        #     'o',
        #     markersize=1,
        #     c='white',
        # )
        
    # plot landmarks as green dots
    ax.scatter(
        X,
        Y,
        s=6,
        linewidth=.5,
        c='lime',
        edgecolors='black',
    )
    
    # plot landmark numbers
    for i in range(len(X)):
        ax.annotate(
            f'{i}',
            (X[i], Y[i]),
            fontsize=6,
        )
    
    fix_axes(X,Y,ax)
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        path = f'./figs/images/{image_set}'
        if not Path(path).exists():
            os.makedirs(path)
        plt.savefig(
            f'{path}/{filename}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()

def scatter(
        title,
        filename,
        df,
        pred,
        target,
        target_name=None,
        alt_name=None,
        save_fig=False,
):
    '''
    This function assumes the predicted values are in the given DataFrame,
    under a column name consisting of the target column name with a '_hat'
    suffix (e.g. target 'baby' requires predictions are in column 'baby_hat').

    Parameters
    ----------
    title : str
        The plot title.
    filename : str
        The filename to save this plot (location is the 'figs' directory).
    df : DataFrame
        The data frame that holds the target, predictors and predictions.
    pred : list of 2 column names
        The X and Y axes of the scatter.
    target : str
        The column to use as target. NOTE: This also establishes the column
        name for predicted values (e.g. target 'baby' requires prediction
        column to be 'baby_hat').
    target_name : str, optional
        Override the target name in the plot legend. The default is None.
    alt_name : str, optional
        Override the display string for target==0 case. The default is None.
    save_fig : bool, optional
        Pass True to save the plot. The default is False.

    Returns
    -------
    None.

    '''
    if target_name is None:
        target_name = target
    if alt_name is None:
        alt_name = f'not {target}'
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(f'{pred[0]}', fontsize=15)
    ax.set_ylabel(f'{pred[1]}', fontsize=15)
    ax.set_title(title)

    t = df[target] == 1
    f = df[target] == 0
    p = df[f'{target}_hat'] == 1
    n = df[f'{target}_hat'] == 0
    tNeg = f & n
    tPos = t & p
    fNeg = t & n
    fPos = f & p

    cases = []
    if np.any(tNeg):
        cases.append(Case(f'True {alt_name}', 'tab:green', tNeg))
    if np.any(tPos):
        cases.append(Case(f'True {target_name}', 'tab:blue', tPos))
    if np.any(fNeg):
        cases.append(Case(f'False {alt_name}', 'tab:red', fNeg))
    if np.any(fPos):
        cases.append(Case(f'False {target_name}', 'tab:orange', fPos))
    for case in cases:
        ax.scatter(
            df[case.mask][pred[0]],
            df[case.mask][pred[1]],
            c=case.color,
            s=50,
            label=case.name
        )
    ax.legend()
    if save_fig:
        plt.savefig(
            f'figs/{filename}',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()
