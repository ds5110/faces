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
        self.baby_meta = self.baby_cache.get_meta('decorated')
        self.adult_meta = self.adult_cache.get_meta('decorated')
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

    def get_image(self, row_id, baby=None):
        '''
        This function does some weird stuff to address different indexes
        for itself (i.e. the combined image metadata) vs its delegates
        (i.e. the original baby face DataFrame in LocalCache and the
        other DataFrame in AltCache).
        
        For example, with the current state of the combined metadata,
        'get_image(693)' returns the same image as 'get_image(4, baby=True)',
        just with a different row_id.
        
        TODO: Rename this function to 'get_anno_img'? This function name
        clashes with the 'get_image' function on the AnnoImg model...
        Also, if we wanted to add a convenience method that would return the
        raw image directly, we would probably want to use this method name.

        Parameters
        ----------
        row_id : int
            This is the row ID or index value for the desired image.
            NOTE: This has different semantics depending on the 'baby'
            argument.
        baby : bool, optional
            If 'baby' is True, then 'row_id' corresponds to the
            index values in the baby DataFrame. If False, then the index
            is relative to the adult faces DataFrame. If 'baby' is not
            explicitly set True nor False, then 'row_id' corresponds to
            the index values in the combined DataFrame. The default is None.

        Returns
        -------
        AnnoImg
            A model with some image metadata and a function to fetch the raw
            image data (as a Pillow image).

        '''
        if baby is None:
            row = self.meta.iloc[row_id]
            if row['baby']:
                # NOTE: This is a hack to get the correct image from the baby
                #       cache, with the correct row ID for the combined data.
                # TODO: Maybe change the index to put babies first, since
                #       we haven't resolved image retrieval for the other
                #       dataset?
                key = row['image_name']
                where_eq = self.baby_meta['image_name'] == key
                baby_row_id = self.baby_meta[where_eq].index[0]
                anno = self.baby_cache.get_image(baby_row_id)
                anno.row_id = row_id
                return anno
        if baby:
            # TODO: Maybe stop supporting this case? Images returned from
            #       this delegate will have row IDs offset by 689 from the
            #       combined DataFrame (i.e. the number of adult faces that
            #       were inserted before the first baby face).
            return self.baby_cache.get_image(row_id)
        else:
            # TODO: Maybe implement the same hack to translate row IDs here?
            #       It's not necessary because the indices should match for
            #       the adult dataset, but those kind of assumptions are
            #       brittle...
            return self.adult_cache.get_image(row_id)

    def save_merged(self):
        '''
        This function is more of a one-time utility. It merges and saves
        (in the data directory) the image metadata from the adult and baby
        datasets and renames all landmark columns to standard names:
         * baby "ground truth" landmarks: 'gt-{dim}{index}' -> '{dim}{index}
         * adult "original" landmarks: 'original_{index}_{dim}' -> '{dim}{index}

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
