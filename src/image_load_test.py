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
import matplotlib.pyplot as plt

base_dir = './data' # assuming cwd is the location of this script
base_url = 'https://coe.northeastern.edu/Research/AClab/InfAnFace/images/'


def load_file(file,url=None,path=base_dir):
    local_path = Path(path)
    local_file = Path(f'{path}/{file}')
    if not url:
        url = f'{base_url}/file'
    if not local_path.exists():
        os.makedirs(local_path)
    elif not local_file.exists():
        with \
                urllib.request.urlopen(url) as infile, \
                open(local_file, 'wb') as outfile:
            outfile.write(infile.read())
            while True:
                data = infile.read(1e5)
                if len(data) < 1: break
                outfile.write(data)
    return local_file

# load the labels data
# NOTE: It's a little awkward, but this _must_ be done between
#       declaring these two functions...
df = pd.read_csv(load_file('labels.csv'))

def get_image(row_id=None,path=None,file=None):
    if row_id is not None:
        path = df['image-set'].iloc[row_id]
        file = df['filename'].iloc[row_id]
    
    image_file = load_file(
        file,
        url=f'{base_url}/{path}/{file}',
        path=f'{base_dir}/images/{path}',
    )
    return Image.open(image_file)

targets = ['turned', 'occluded', 'tilted', 'expressive']
x_cols = [col for col in df if col.startswith('gt-x')]
y_cols = [col for col in df if col.startswith('gt-y')]

def to_image(series):
    return get_image(
        path=series['image-set'],
        file=series['filename'],
    )

def plot_image(row_id=None,series=None):
    if not series:
        series = df.iloc[0,:]
    img = to_image(series)
    plt.imshow(img)
    plt.scatter(
        series[x_cols],
        series[y_cols],
        s=10,
        linewidth=.5,
        c='lightgreen',
        edgecolors='black',
    )
    plt.show()

plot_image(0)

#-- scrape all images
# for i in range(df.shape[0]):
#     get_image(i)

print(f'target counts:\n{df.loc[:,targets].sum()}\n')

no_targets = reduce(iand, [df[col] == 0 for col in targets])
print(f'no targets:  {df[no_targets].shape}')
print(f'one or more: {df[~no_targets].shape}')
