#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import pandas as pd

# project
from util.local_cache import LocalCache
from util.alt_cache import AltCache
from util.column_names import landmark_cols, alt_cols, merged_cols

data_path = 'data'
meta_filename = 'merged_landmarks.csv'
default_baby = LocalCache()
default_adult = AltCache()

# dictionaries for merging landmarks
from_baby = {landmark_cols[i]: merged_cols[i] for i in range(len(merged_cols))}
from_adult = {alt_cols[i]: merged_cols[i] for i in range(len(merged_cols))}

class MetaCache:
    def __init__(
            self,
            baby_cache=default_baby,
            adult_cache=default_adult,
    ):
        self.baby_cache = baby_cache
        self.adult_cache = adult_cache
        self.meta = pd.read_csv(
            f'{data_path}/{meta_filename}',
            dtype={
                'image-set': str,
                'filename': str,
                'partition': str,
                'subpartition': str,
            }
        )

    def get_meta(self, alt=None):
        if alt is not None:
            if alt.startswith('baby'):
                if alt == 'baby_raw':
                    return self.baby_cache.get_meta()
                else:
                    return self.baby_cache.get_meta('decorated')
            if alt.startswith('adult'):
                if alt == 'adult_raw':
                    return self.adult_cache.get_meta()
                else:
                    return self.adult_cache.get_meta('decorated')
        return self.meta.copy()

    def get_image(self, row_id=0, baby=None):
        if baby is None:
            baby = self.meta.loc[row_id]['baby']
        if baby:
            return self.baby_cache.get_image(row_id)
        else:
            return self.adult_cache.get_image(row_id)

    def save_merged(self):
        '''
        This function is more of a one-time utility. It merges and saves
        (in the data directory) the image metadata from the adult and baby
        datasets and renames all landmark columns to standard names:
         * baby "ground truth" landmarks: 'gt-{dim}{index}' -> '{dim}{index}
         * adult "original" landmakrs: 'original_{index}_{dim}' -> '{dim}{index}

        Returns
        -------
        None.

        '''
        dfs = []
        for i, (cache, renames) in enumerate([
                (default_adult, from_adult),
                (default_baby, from_baby),
        ]):
            df = cache.get_meta('decorated').rename(columns=renames)
            df['baby'] = i
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(f'{data_path}/{meta_filename}', index=False)
