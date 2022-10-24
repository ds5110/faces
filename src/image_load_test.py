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

base_dir = '../data' # assuming cwd is the location of this script
base_url = 'https://coe.northeastern.edu/Research/AClab/InfAnFace/images/'

df = pd.read_csv(f'{base_dir}/labels.csv')

def get_image(row_id=None,path=None,file=None):
    '''
    This is a simple function for pulling raw image data to a local cache.

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
    TYPE
        DESCRIPTION.

    '''
    if row_id is not None:
        path = df['image-set'].iloc[row_id]
        file = df['filename'].iloc[row_id]
    
    local_path = Path(f'{base_dir}/{path}')
    local_file = Path(f'{local_path}/{file}')
    if not local_path.exists():
        print('making image-set directory...')
        os.makedirs(local_path)
        print('OK')
    elif not local_file.exists():
        print('making local copy..')
        with urllib.request.urlopen(f'{base_url}/{path}/{file}') as infile, \
             open(local_file, 'wb') as outfile:
            outfile.write(infile.read())
            while True:
                info = infile.read(1e5)
                if len(info) < 1: break
                outfile.write(info)
        print('OK')
    return Image.open(local_file)

get_image(0).show()