#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 17:21:41 2022

@author: jhautala
"""

import os
from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
from PIL import Image

from util.model import AnnoImg, x_cols, y_cols

class LocalCache:
    def __init__(
            self,
            base_dir='data',
            scratch_dir='scratch',
            base_url='https://coe.northeastern.edu/Research/AClab/InfAnFace',
    ):
        self.base_dir = base_dir
        self.scratch_dir = scratch_dir
        self.base_url = base_url
        self.meta = pd.read_csv(self.get_file('labels.csv'))
    
    def get_meta(self,desc=None):
        '''
        To help avoid side-effects, this method creates a new copy
        of the image metadata for each call.

        Returns
        -------
        DataFrame
            A DataFrame containing all the InfAnFace metadata.

        '''
        if desc is not None:
            filepath = f'{self.scratch_dir}/labels_{desc}.csv'
            return pd.read_csv(filepath)
        
        return self.meta.copy()
    
    def save_meta(self,df,desc):
        df.to_csv(
            f'{self.scratch_dir}/labels_{desc}.csv',
            index=False,
        )
    
    def get_file(self,file,url=None,local_path=None):
        local_path = Path(local_path or self.base_dir)
        local_file = Path(f'{local_path}/{file}')
        if not url:
            url = f'{self.base_url}/{file}'
        if not local_path.exists():
            print(f'creating missing directory: {local_path}')
            os.makedirs(local_path)
        if not local_file.exists():
            print(f'downloading missing file: {local_path}/{local_file}')
            with \
                    urllib.request.urlopen(url) as infile, \
                    open(local_file, 'wb') as outfile:
                outfile.write(infile.read())
                while True:
                    data = infile.read(1e5)
                    if len(data) < 1: break
                    outfile.write(data)
        return local_file

    def get_local(self,file):
        return f'{self.base_dir}/{file}'
    
    def get_image(self,row_id,desc=None):
        path = self.meta['image-set'].iloc[row_id]
        file = self.meta['filename'].iloc[row_id]
        
        image_file = self.get_file(
            file,
            url=f'{self.base_url}/images/{path}/{file}',
            local_path=f'{self.base_dir}/images/{path}',
        )
        coords = np.stack(
            [self.meta[cols].loc[row_id,:].values for cols in [x_cols, y_cols]],
            1
        )
        desc = [desc] if isinstance(desc, str) else desc
        return AnnoImg(
            path,
            file,
            coords,
            lambda: Image.open(image_file),
            row_id=row_id,
            desc=desc,
        )

cache = LocalCache() # default cache
