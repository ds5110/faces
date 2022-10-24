#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:51:12 2022

@author: jhautala
"""

import os
from pathlib import Path
import urllib.request
import pandas as pd
from PIL import Image
from operator import iand
from functools import reduce
import matplotlib.pyplot as plt

base_dir = '../data' # assuming cwd is the location of this script
base_url = 'https://coe.northeastern.edu/Research/AClab/InfAnFace/images/'


def load_file(file,url=None,path=base_dir):
    local_path = Path(path)
    local_file = Path(f'{path}/{file}')
    if not url:
        url = f'{base_url}/file'
    if not local_path.exists():
        os.makedirs(local_path)
    elif not local_file.exists():
        with \
                urllib.request.urlopen(url) as infile, \
                open(local_file, 'wb') as outfile:
            outfile.write(infile.read())
            while True:
                data = infile.read(1e5)
                if len(data) < 1: break
                outfile.write(data)
    return local_file

# load the labels data
# NOTE: It's a little awkward, but this _mus be done between declaring these
#       two functions...
df = pd.read_csv(load_file('labels.csv'))

def get_image(row_id=None,path=None,file=None):
    '''
    This is a simple function for pulling raw image data to a local cache.
    You can access images by either their 'path' and 'file' (corresponding to
    'image-set' and 'filename' in the 'labels' DataFrame) or by row ID
    (the index in the 'labels' DataFrame)

    Parameters
    ----------
    row_id : integer, optional
        DESCRIPTION. The default is None.
    path : TYPE, optional
        DESCRIPTION. The default is None.
    file : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    A pillow Image

    '''
    if row_id is not None:
        path = df['image-set'].iloc[row_id]
        file = df['filename'].iloc[row_id]
    
    image_file = load_file(
        file,
        f'{base_url}/{path}/{file}',
        f'{base_dir}/{path}',
    )
    return Image.open(image_file)

def to_image(series):
    return get_image(path=series['image-set'],file=series['filename'])

# e.g. usage
# plt.imshow(get_image(1))

# e.g. per row
im_df = df.iloc[0,:]
im = to_image(im_df)
plt.imshow(im)

targets = ['turned', 'occluded', 'tilted', 'expressive']
print(f'target counts:\n{df.loc[:,targets].sum()}\n')

no_targets = reduce(iand, [df[col] == 0 for col in targets])
print(f'no targets:  {df[no_targets].shape}')
print(f'one or more: {df[~no_targets].shape}')
