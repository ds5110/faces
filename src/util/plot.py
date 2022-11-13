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
        scatter_points=None,
        ref_point=None,
        annotate=None,
        cross=False,
        grayscale=False,
        skip_img=False,
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
    
    # anno=rotated
    # annotate='scatternum'
    # annotate='spline'
    # ref_point=mid
    # cross=True
    # grayscale=True
    # save_fig=False
    image_set = anno.image_set
    filename = anno.filename
    desc = f' (row {anno.row_id})' if anno.row_id is not None else ''
    if len(anno.desc):
        desc += f' ({", ".join(anno.desc)})'
    title = f'{image_set}/{filename}' + desc
    X = anno.get_x()
    Y = anno.get_y()
    img = anno.get_image()
    if grayscale:
        # NOTE: Extracting luma component produces the same result
        #       as ImageOps.grayscale.
        # y,u,v = img.convert('YCbCr').split()
        img = ImageOps.grayscale(img)
    
    # get the image data
    # img.convert('YCbCr')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    if not skip_img:
        ax.imshow(img)
    if cross:
        center= np.array([img.width/2, img.height/2])
        ax.axhline(y=center[1])
        ax.axvline(x=center[0])
    
    if annotate:
        if annotate.startswith('spline'):
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
        if anno.desc is not None:
            filename += f'_({"_".join(anno.desc)})'
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
