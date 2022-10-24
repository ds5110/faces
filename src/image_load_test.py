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

base_dir = '../data' # assuming cwd is the location of this script
base_url = 'https://coe.northeastern.edu/Research/AClab/InfAnFace/images/'

df = pd.read_csv(f'{base_dir}/labels.csv')

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
    
    local_path = Path(f'{base_dir}/{path}')
    local_file = Path(f'{local_path}/{file}')
    if not local_path.exists():
        os.makedirs(local_path)
    elif not local_file.exists():
        with urllib.request.urlopen(f'{base_url}/{path}/{file}') as infile, \
             open(local_file, 'wb') as outfile:
            outfile.write(infile.read())
            while True:
                info = infile.read(1e5)
                if len(info) < 1: break
                outfile.write(info)
    return Image.open(local_file)

# e.g. usage
# get_image(0).show()

targets = ['turned', 'occluded', 'tilted', 'expressive']
print(df.loc[:,targets].sum())

no_targets = reduce(lambda x, y: x & y, [df[col] == 0 for col in targets])
print(f'no targets:  {df[no_targets].shape}')
print(f'one or more: {df[~no_targets].shape}')
