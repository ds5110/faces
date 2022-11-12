# -*- coding: utf-8 -*-

"""
This script decorates the original metadata DataFrame, adding some
geometric information derived from landmark data.

@author: jhautala
"""

import numpy as np
import pandas as pd

# intra-project
from util.local_cache import cache
from util.pre import get_rotate_data



if __name__ == '__main__':
    # load the labels data
    df = cache.get_meta()
    
    widths = []
    heights = []
    angles = []
    cenrots = []
    raws = []
    faces = []
    
    # calculate rotation/etc for each image
    for i in range(df.shape[0]):
        anno = cache.get_image(i)
        raws.append(anno.get_coords())
        width, height, face, angle, cenrot = get_rotate_data(anno)
        faces.append(face)
        widths.append(width)
        heights.append(height)
        angles.append(angle)
        cenrots.append(cenrot)

    # add image dimensions
    df['width'] = widths
    df['height'] = heights
    
    # add angle of rotation (in radians)
    df['yaw'] = angles
    
    # add normalized landmarks
    raws = np.array(raws)
    mins = np.amin(raws,axis=1)
    maxs = np.amax(raws,axis=1)
    extents = maxs-mins
    for i in range(raws.shape[1]):
        tmp_df = pd.DataFrame(
            data=(raws[:,i,:] - faces)/extents,
            columns=[f'norm-{dim}{i}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # add centered/rotated landmarks
    cenrots = np.array(cenrots)
    for i in range(cenrots.shape[1]):
        tmp_df = pd.DataFrame(
            data=cenrots[:,i,:],
            columns=[f'cenrot-{dim}{i}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # calculate extents of landmarks
    mins = np.amin(cenrots,axis=1)
    maxs = np.amax(cenrots,axis=1)
    extents = maxs-mins
    
    # calculate centers
    all_dims = np.stack([widths,heights]).T
    centers = all_dims/2.
    
    # add normalized landmarks (centered, rotated and scaled per extent)
    for i in range(cenrots.shape[1]):
        tmp_df = pd.DataFrame(
            data=(cenrots[:,i,:] - centers)/extents,
            columns=[f'norm_cenrot-{dim}{i}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    cache.save_meta(df,'decorated')
