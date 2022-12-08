#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:40:12 2022

@author: jhautala
"""

# 3rd party
import numpy as np

# internal
from util.local_cache import LocalCache
from util.model import AnnoImg
from util.column_names import alt_x_cols, alt_y_cols 


class AltCache(LocalCache):
    def __init__(
            self,
            base_dir='_data',
            base_url='https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/raw/master/data/300w',
            meta_filename='300w_valid.csv',
    ):
        super().__init__(base_dir,base_url,meta_filename)

    def get_image(self, row_id, desc=None):
        elems = self.meta['image_name'].iloc[row_id].split('/')
        path = '/'.join(elems[:-1])
        file = elems[-1]

        coords = np.stack(
            [self.meta[cols].loc[row_id, :].values for cols in [alt_x_cols, alt_y_cols]],
            1
        )
        desc = [desc] if isinstance(desc, str) else desc
        return AnnoImg(
            path,
            file,
            coords,
            lambda: None,
            row_id=row_id,
            desc=desc,
        )
