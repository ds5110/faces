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

# project
from util.model import AnnoImg
from util.column_names import x_cols, y_cols
from util.pre import add_derived


class LocalCache:
    def __init__(
            self,
            base_dir='_data',
            base_url='https://coe.northeastern.edu/Research/AClab/InfAnFace',
            meta_filename='labels.csv',
    ):
        self.base_dir = base_dir
        self.base_url = base_url
        self.meta_filename = meta_filename
        self.meta = pd.read_csv(self.get_file(meta_filename))
    
    def get_meta(self, desc=None):
        '''
        To avoid side-effects, this method creates a new copy
        of the image metadata for each call.

        Returns
        -------
        DataFrame
            A DataFrame containing all the InfAnFace metadata.

        '''
        if desc is not None:
            elems = self.meta_filename.split('.')
            filename = f'{self.base_dir}/{elems[0]}_{desc}.{elems[1]}'
            if not Path(filename).exists():
                df = add_derived(self)
                df.to_csv(filename, index=False)
            return pd.read_csv(filename)
        
        return self.meta.copy()
    
    def save_meta(self, df, desc):
        elems = self.meta_filename.split('.')
        filename = f'{self.base_dir}/{elems[0]}_{desc}.{elems[1]}'
        df.to_csv(
            filename,
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
            print(f'downloading missing file: {local_file}')
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
