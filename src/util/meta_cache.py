#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jhautala
"""

import pandas as pd

# project
from .local_cache import LocalCache
from .alt_cache import AltCache


class MetaCache:
    def __init__(self, baby_cache=AltCache(), adult_cache=LocalCache()):
        self.baby_cache = baby_cache
        self.adult_cache = adult_cache
        df_baby = self.baby_cache.get_meta('decorated')
        df_adult = self.adult_cache.get_meta('decorated')
        df_baby['baby'] = 1
        df_adult['baby'] = 0
        self.meta = pd.concat([df_baby, df_adult], ignore_index=True)

    def get_meta(self):
        return self.meta.copy()

    def get_image(self, row_id=0, baby=None):
        if baby is None:
            baby = self.meta.loc[row_id]['baby']
        if baby:
            return self.baby_cache.get_image(row_id)
        else:
            return self.adult_cache.get_image(row_id)