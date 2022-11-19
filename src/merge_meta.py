#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is yet another somewhat pragmatic one-time utility. It merges the
image metadata from the adult and baby datasets and renames all landmark
columns to standard names:
 * baby "ground truth" landmarks: 'gt-{dim}{index}' -> '{dim}{index}
 * adult "original" landmakrs: 'original_{index}_{dim}' -> '{dim}{index}

@author: jhautala
"""

import pandas as pd

# project
from util.meta_cache import default_baby, default_adult, meta_filename

# constants for remapping landmark columns
merged_cols = []
for cols in [[f'{axis}{i}' for i in range(68)] for axis in ['x', 'y']]:
    merged_cols.extend(cols)

baby_cols = []
for cols in [[f'gt-{axis}{i}' for i in range(68)] for axis in ['x', 'y']]:
    baby_cols.extend(cols)
from_baby = {baby_cols[i]: merged_cols[i] for i in range(len(merged_cols))}

adult_cols = []
for cols in [[f'original_{i}_{axis}' for i in range(68)] for axis in ['x', 'y']]:
    adult_cols.extend(cols)
from_adult = {adult_cols[i]: merged_cols[i] for i in range(len(merged_cols))}

# execution to merge data
dfs = []
for i, (cache, renames) in enumerate([
        (default_adult, from_adult),
        (default_baby, from_baby),
]):
    df = cache.get_meta('decorated').rename(columns=renames)
    df['baby'] = i
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df.to_csv(f'data/{meta_filename}', index=False)
