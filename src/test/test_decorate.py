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
all_coords = []
if __name__ == '__main__':
    for i in range(df.shape[0]):
        anno = cache.get_image(i)
        width, height, face, angle, coords = get_rotate_data(anno)
        widths.append(width)
        heights.append(height)
        angles.append(angle)
        all_coords.append(coords)

# add angle of rotation (in radians)
df['yaw'] = angles

all_coords = np.array(all_coords)
for i in range(all_coords.shape[1]):
    tmp_df = pd.DataFrame(
        data=all_coords[:,i,:],
        columns=[f'cenrot-{dim}{i}' for dim in ['x','y']]
    )
    df = pd.concat([df,tmp_df], axis=1)

df['width'] = widths
df['height'] = heights
all_dims = np.stack([widths,heights]).T

for i in range(all_coords.shape[1]):
    tmp_df = pd.DataFrame(
        data=all_coords[:,i,:]/all_dims - .5,
        columns=[f'norm_cenrot-{dim}{i}' for dim in ['x','y']]
    )
    df = pd.concat([df,tmp_df], axis=1)

cache.save_meta(df,'decorated')