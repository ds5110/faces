#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 19:25:22 2022

@author: jhautala
"""

import numpy as np
import pandas as pd

from util.local_cache import cache
from util.model import norm_cols

df = cache.get_meta('decorated')

# expect extents to add up to 1
for i in range(2):
    mins = np.amin(np.array(df[norm_cols[i]].values), axis=1)
    maxs = np.amax(np.array(df[norm_cols[i]].values), axis=1)
    print(f'\nchecking {"x" if i == 0 else "y"} extent:')
    print(pd.DataFrame(np.abs(mins)+maxs).describe())

#-- e.g. calculate image centers from decorated df
centers = np.array(df[['width','height']].values)/2.
