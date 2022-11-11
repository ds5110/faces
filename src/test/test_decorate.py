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

all_cen = []
all_ang = []
all_coo = []
if __name__ == '__main__':
    for i in range(df.shape[0]):
        anno = cache.get_image(i)
        face_cen, angle, coords = get_rotate_data(anno)
        all_cen.append(face_cen)
        all_ang.append(angle)
        all_coo.append(coords)

df['yaw'] = all_ang

all_cen = np.array(all_cen)
df['center-x'] = all_cen[:,0]
df['center-y'] = all_cen[:,1]

all_coo = np.array(all_coo)
for i in range(all_coo.shape[1]):
    tmp_df = pd.DataFrame(
        data=all_coo[:,i,:],
        columns=[f'cenrot-{dim}{i}' for dim in ['x','y']]
    )
    df = pd.concat([df,tmp_df], axis=1)


# TODO figure out a more elegant way?
for dim in ['x','y']:
    for col in df.columns:
        if col.startswith(f'cenrot-{dim}'):
            df[f'norm_{col}'] = df[col] - df[f'center-{dim}']

cache.save_meta(df,'decorated')