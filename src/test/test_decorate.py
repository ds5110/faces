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

# load the labels data
df = cache.get_meta()

widths = []
heights = []
angles = []
cenrots = []
if __name__ == '__main__':
    for i in range(df.shape[0]):
        anno = cache.get_image(i)
        width, height, face, angle, coords = get_rotate_data(anno)
        widths.append(width)
        heights.append(height)
        angles.append(angle)
        cenrots.append(coords)

# add angle of rotation (in radians)
df['yaw'] = angles

# add centered/rotated coordinates
cenrots = np.array(cenrots)
for i in range(cenrots.shape[1]):
    tmp_df = pd.DataFrame(
        data=cenrots[:,i,:],
        columns=[f'cenrot-{dim}{i}' for dim in ['x','y']]
    )
    df = pd.concat([df,tmp_df], axis=1)

# calculate extents of landmarks
x_coords = cenrots[:,:,0]
min_x = np.amin(x_coords,axis=1)
max_x = np.amax(x_coords,axis=1)

y_coords = cenrots[:,:,1]
min_y = np.amin(y_coords,axis=1)
max_y = np.amax(y_coords,axis=1)
extents = np.stack([max_x - min_x, max_y - min_y]).T

# extract width/height and calculate centers
df['width'] = widths
df['height'] = heights
all_dims = np.stack([widths,heights]).T
centers = all_dims/2.

for i in range(cenrots.shape[1]):
    tmp_df = pd.DataFrame(
        data=(cenrots[:,i,:] - centers)/extents,
        columns=[f'norm_cenrot-{dim}{i}' for dim in ['x','y']]
    )
    df = pd.concat([df,tmp_df], axis=1)

cache.save_meta(df,'decorated')


#-- e.g. calculate image centers from decorated df
# np.array(df[['width','height']].values)/2.

