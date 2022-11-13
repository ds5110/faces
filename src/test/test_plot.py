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
from util.pre import rotate, crop, cheeks
from util.model import cat_cols, cenrot_cols

# load the labels data
df = cache.get_meta('decorated')

def annotated_plot(types=None,save_fig=False,n=10):
    if types is None:
        types = [None,'scatter','scatternum','spline','splinelabel']
    for i in range(n):
        anno = cache.get_image(i)
        for annotate in types:
            plot_image(
                anno,
                annotate=annotate,
                cross=False,
                save_fig=save_fig,
            )
            rotated = rotate(anno)
            plot_image(
                rotated,
                annotate=annotate,
                cross=True,
                save_fig=save_fig,
            )
            cropped = crop(rotated)
            plot_image(
                cropped,
                annotate=annotate,
                cross=True,
                save_fig=save_fig,
            )

def plot_stuff(
        anno,
        annotate='spline',
        save_fig=False,
):
    plot_image(
        anno,
        # annotate=annotate,
        cross=False,
        save_fig=save_fig,
    )
    
    rotated = rotate(anno)
    plot_image(
        rotated,
        annotate=annotate,
        cross=True,
        grayscale=True,
        save_fig=save_fig,
    )
    
    # plot geometry without image, to verify coords
    plot_image(
        rotated,
        annotate=annotate,
        cross=True,
        grayscale=True,
        skip_img=True,
        save_fig=save_fig,
    )
    
    # # plot coords
    # row_id = anno.row_id
    # coords = np.stack(
    #     [df[cols].loc[row_id,:].values for cols in cenrot_cols],
    #     1
    # )
    # plot_coords(
    #     coords,
    #     df.loc[row_id,:]['cenrot_width'],
    #     df.loc[row_id,:]['cenrot_height'],
    #     save_fig=save_fig,
    # )
    
    # # try simple crop
    # cropped = crop(rotated)
    # plot_image(
    #     cropped,
    #     annotate=annotate,
    #     cross=True,
    #     grayscale=True,
    #     # skip_img=True,
    #     save_fig=save_fig,
    # )
    
    # try crop to splines
    cropped = crop(rotated,use_splines=True)
    plot_image(
        cropped,
        annotate=annotate,
        cross=True,
        grayscale=True,
        # skip_img=True,
        save_fig=save_fig,
    )

def test_challenging(
        code=None,
        all_occurrences=False,
        save_fig=False,
):
    '''
    loop over images per categorical combinations

    Parameters
    ----------
    code : int from [0,16], optional
        DESCRIPTION. The default is None.
    all_occurrences : bool, optional
        whether to loop over all occurrences per combo. The default is False.
    save_fig : bool, optional
        whether to save the output. The default is False.

    Returns
    -------
    None.

    '''
    codes = range(1,16) if code is None else [code]
    masks = [df[col] == 1 for i, col in enumerate(cat_cols)]
    for combo in [f'{i:b}'.rjust(4,'0') for i in codes]:
        desc = [col for i, col in enumerate(cat_cols) if combo[i] == '1']
        mask = [m if combo[i] == '1' else ~m for i, m in enumerate(masks)]
        tmp = df[reduce(lambda a,b: a & b, mask)]
        if tmp.shape[0] == 0:
            print(f'no rows found for combination! ({desc})')
            continue;
        print(f'found {tmp.shape[0]} rows ({desc})')
        for i in range(tmp.shape[0]):
            row_id = tmp.index[i]
            anno = cache.get_image(row_id,desc)
            plot_stuff(anno)
    
def plot_row_ids(row_ids,desc=None):
    suff = f' (desc: {desc})' if desc is not None else ''
    for row_id in row_ids:
        print(f'plotting {row_id}{suff}')
        plot_stuff(
            cache.get_image(row_id,desc),
        )

def plot_extreme_diffs(save_fig=False):
    # col =[col for col in df.columns if col.startswith('rot_sym_diff-x')][0]
    for col in [col for col in df.columns if col.startswith('rot_sym_diff-')]:
        anno_min = cache.get_image(df[col].idxmin(),f'min {col}')
        anno_max = cache.get_image(df[col].idxmax(),f'max {col}')
        plot_stuff(anno_min)
        plot_stuff(anno_max)

def plot_crossing():
    '''
    check for cheek points crossing the centerline (center of nose)

    Returns
    -------
    None.

    '''
    row_ids = set()
    for i in cheeks[:,1]:
        col = f'cenrot-x{i}'
        tmp = df[df[col] <= df['cenrot-x33']]
        for i in range(tmp.shape[0]):
            row_id = tmp.index[i]
            row_ids.add(row_id)
    plot_row_ids(sorted(list(row_ids)),'left_cheek')

    row_ids = set()
    for i in cheeks[:,0]:
        col = f'cenrot-x{i}'
        tmp = df[df[col] >= df['cenrot-x33']]
        for i in range(tmp.shape[0]):
            row_id = tmp.index[i]
            row_ids.add(row_id)
    plot_row_ids(sorted(list(row_ids)),'right_cheek')

if __name__ == '__main__':
    #------ basic usage
    # annotated_plot()
    # annotated_plot(['spline'])
    test_challenging(11)
    # test_challenging(all_occurrences=True)
    # plot_crossing()
    #-- plot all images
    # plot_row_ids(range(df.shape[0]))
    
    #------ troubleshooting junk
    # #-- spline issues
    # spline_warn = [
    #     101,
    #     159,
    #     288,
    #     348,
    # ]
    
    #-- cenrot issue (content out of frame)
    # crop_issues = [
    #     118,
    # ]
    # plot_row_ids(crop_issues,'crop_issues')
    
    #-- junk around testing yaw caclulation
    # rot_fail = [
    #     80,
    #     # 81, # ok
    #     # 99, # ok, just weird
    #     141,
    #     286,
    #     319,
    #     239,
    # ]
    # rot_fail = [
    #     # 49,
    #     237,
    # ]
    # rot_fail = [
    #     49,
    #     80,
    #     81,
    #     99,
    #     141,
    #     286,
    #     319,
    #     237,
    #     239,
    # ]
    # plot_row_ids(rot_fail,'rot_fail')
    # plot_row_ids(range(df.shape[0]))
    # plot_row_ids([6])
    
    #-- other one-off stuff
    # cute_crop = [
    #     387,
    # ]
    # plot_stuff(
    #     cache.get_image(80),
    #     annotate='scatternum',
    #     save_fig=True
    # )
    # plot_image(
    #     rotate(cache.get_image(80)),
    #     annotate='scatternum',
    #     points=[36, 45, 39, 42],
    #     cross=True,
    #     save_fig=True,
    # )
