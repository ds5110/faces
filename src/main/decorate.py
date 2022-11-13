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
from util.pre import get_yaw_data, h_syms, v_line, cheeks
from util.model import nose_i


if __name__ == '__main__':
    # load the labels data
    df = cache.get_meta()
    
    widths = []
    heights = []
    margins = [] 
    angles = []
    cenrots = []
    raws = []
    faces = []
    cenrot_widths = []
    cenrot_heights = []
    
    # calculate rotation/etc for each image
    for i in range(df.shape[0]):
        anno = cache.get_image(i)
        raws.append(anno.get_coords())
        in_dims, margin, face, angle, cenrot = get_yaw_data(anno)
        faces.append(face)
        widths.append(in_dims[0])
        heights.append(in_dims[1])
        margins.append(margins)
        cenrot_widths.append(widths[-1] + margin[0])
        cenrot_heights.append(heights[-1] + margin[1])
        angles.append(angle)
        cenrots.append(cenrot)
    
    # convert to numpy
    raws = np.array(raws)
    cenrots = np.array(cenrots)
    
    # add image dimensions
    df['width'] = widths
    df['height'] = heights
    df['cenrot_width'] = cenrot_widths
    df['cenrot_height'] = cenrot_heights
    
    # add angle of rotation (in radians)
    df['yaw'] = angles
    
    # roll estimate
    mid_y = cenrots[cheeks[0,0],1]
    cheeks_x = cenrots[:,cheeks.ravel(),0]
    min_x = np.min(cheeks_x,axis=1)
    radius = (np.max(cheeks_x,axis=1) - min_x)/2
    mid_x = min_x + radius
    df['roll'] = np.arctan((cenrots[:,nose_i,0]-mid_x)/radius)
    
    for col in ['yaw','roll']:
        df[f'{col}_abs'] = np.abs(df[col])
    
    # add normalized landmarks
    # this is the distance from nose as a proportion of landmarks' extents
    mins = np.amin(raws,axis=1)
    maxs = np.amax(raws,axis=1)
    extents = maxs-mins
    for i in range(raws.shape[1]):
        tmp_df = pd.DataFrame(
            data=(raws[:,i,:] - faces)/extents,
            columns=[f'norm-{dim}{i}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # calculate distance between expected symmetric points (raw)
    for i, [p1, p2] in enumerate(h_syms):
        tmp_df = pd.DataFrame(
            data=np.squeeze(np.diff(raws[:,[p1,p2],:],axis=1)),
            columns=[f'sym_diff-{dim}{p1}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # add centered/rotated landmarks
    for i in range(cenrots.shape[1]):
        tmp_df = pd.DataFrame(
            data=cenrots[:,i,:],
            columns=[f'cenrot-{dim}{i}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # calculate extents of landmarks (rotated)
    mins = np.amin(cenrots,axis=1)
    maxs = np.amax(cenrots,axis=1)
    extents = maxs-mins
    
    # calculate centers
    all_dims = np.stack([widths,heights]).T
    centers = all_dims/2.
    
    # add normalized landmarks (centered, rotated and scaled per extent)
    for i in range(cenrots.shape[1]):
        tmp_df = pd.DataFrame(
            data=(cenrots[:,i,:] - cenrots[:,nose_i,:])/extents,
            columns=[f'norm_cenrot-{dim}{i}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # calculate distance between expected symmetric points (rotated)
    for i, [p1, p2] in enumerate(h_syms):
        tmp_df = pd.DataFrame(
            data=np.squeeze(np.diff(cenrots[:,[p1,p2],:],axis=1)),
            columns=[f'cenrot_sym_diff-{dim}{p1}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    # calculate normalized distance between expected symmetric points (rotated)
    for i, [p1, p2] in enumerate(h_syms):
        tmp_df = pd.DataFrame(
            data=np.squeeze(np.diff(cenrots[:,[p1,p2],:],axis=1))/extents,
            columns=[f'norm_cenrot_sym_diff-{dim}{p1}' for dim in ['x','y']]
        )
        df = pd.concat([df,tmp_df], axis=1)
    
    cache.save_meta(df,'decorated')
