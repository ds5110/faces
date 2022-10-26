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
import numpy as np
from PIL import Image
from operator import iand
from functools import reduce
import matplotlib.pyplot as plt
import re
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

class Feature:
    def __init__(self, desc, idx):
        self.desc = desc
        self.idx = idx

class LandmarkModel:
    def __init__(self,features):
        self.features = features
        self.per_desc = {f.desc for f in features}

landmark68 = LandmarkModel(
    [
        Feature('right_cheek', range(8)), # overlaps chin by 1
        Feature('chin', range(7,10)), # center at 8
        Feature('left_cheek', range(9,17)), # overlaps chin by 1
        Feature('right_brow', range(17,22)),
        Feature('left_brow', range(22,27)),
        Feature('nose_v', range(27,31)),
        Feature('nose_h', range(31,36)),
        Feature('right_eye_top', range(36,40)),
        Feature('right_eye_bot', [*range(39,42),36]),
        Feature('left_eye_top', range(42,46)),
        Feature('left_eye_bot', [*range(45,48),42]),
        Feature('lip_out_top_right', range(48,51)), # overlaps philtrum by 1
        Feature('philtrum', range(50,53)), # center at 51
        Feature('lip_out_top_left', range(52,55)), # overlaps philtrum by 1
        Feature('lip_out_bot', range(55,60)),
        Feature('lip_in_top', range(60,65)),
        Feature('lip_in_bot', [*range(64,68),60]),
    ]
)

class LocalCache:
    def __init__(
            self,
            base_dir='./data',
            base_url='https://coe.northeastern.edu/Research/AClab/InfAnFace'
    ):
        self.base_dir = base_dir
        self.base_url = base_url
        self.meta = pd.read_csv(self.get_file('labels.csv'))
    
    def get_meta(self):
        return self.meta.copy()
    
    def get_file(self,file,url=None,local_path=None):
        print(f'trying to get file "{file}" from url "{url} "to local "{local_path}"')
        local_path = Path(local_path or self.base_dir)
        local_file = Path(f'{local_path}/{file}')
        if not url:
            url = f'{self.base_url}/{file}'
            print(f'derived url: {url}')
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

    def get_local(self,file):
        return f'{self.base_dir}/{file}'
    
    def get_image(self,row_id=None,path=None,file=None):
        if row_id is not None:
            path = self.meta['image-set'].iloc[row_id]
            file = self.meta['filename'].iloc[row_id]
        
        image_file = self.get_file(
            file,
            url=f'{self.base_url}/images/{path}/{file}',
            local_path=f'{self.base_dir}/images/{path}',
        )
        return Image.open(image_file)


class ImagePlus:
    def __init__(self,img,model):
        self.img = img
        self.model = model

# load the labels data
# NOTE: It's a little awkward, but this _must_ be done between
#       declaring these two functions...
data = LocalCache()
df = data.get_meta()
target_cols = ['turned', 'occluded', 'tilted', 'expressive']
x_cols = [col for col in df if col.startswith('gt-x')]
y_cols = [col for col in df if col.startswith('gt-y')]

other = pd.read_csv(data.get_local('300w_infanface_train.csv'))
# other.dtypes

def plot_image(
        row_id=None,
        series=None,
        annotate=None,
        save_fig=False
):
    if not series:
        series = df.iloc[row_id,:]
    category = series['image-set']
    filename = series['filename']
    title = f'{category}/{filename}'
    if row_id is not None:
        title += f' (row {row_id})'
    
    # get the image data
    img = data.get_image(
        path=series['image-set'],
        file=series['filename'],
    )
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    
    if annotate:
        if annotate.startswith('spline'):
            for f in landmark68.features:
                # f = f_per_desc['left_eye_top']# problematic for 3-spline
                
                # print('trying feature: ' + f.desc)
                xx = series[[f'gt-x{i}' for i in f.idx]].astype(float).values
                yy = series[[f'gt-y{i}' for i in f.idx]].astype(float).values
                if pd.isna(xx).any(): continue;
                if pd.isna(yy).any(): continue;
                
                points = np.stack((xx,yy))
                # print(points.shape)
                distance = np.cumsum(
                    np.sqrt(np.sum(
                        np.diff(points.T, axis=0)**2,
                        axis=1
                    ))
                )
                # print(distance.shape)
                if not distance[-1]: continue;
                
                # print('x', xx)
                # print('y', yy)
                # print('pre distance', distance)
                distance = np.insert(distance, 0, 0)/distance[-1]
                # print('post distance', distance)
                splines = [UnivariateSpline(distance, point, k=2, s=.2) for point in points]
                points_fitted = np.vstack(
                    [spline(np.linspace(0, 1, 64)) for spline in splines]
                )
                # if points_fitted
                
                # plt.plot(*points, 'ok', label='original points')
                plt.plot(
                    *points_fitted,
                    linestyle='-',
                    linewidth='1',
                    c='fuchsia',
                )
                
                if 'splinelabel' == annotate:
                    # TODO: come up with a way to avoid overlapping labels
                    ax.annotate(
                        f'{f.desc}',
                        (
                            xx[0],
                            yy[0]
                            # xx[np.argmax(xx)],
                            # yy[np.argmin(xx)]
                            # np.mean(xx),
                            # np.mean(yy)
                        ),
                        fontsize=6,
                    )
        if annotate and annotate.startswith('scatter'):
            ax.scatter(
                series[x_cols],
                series[y_cols],
                s=6,
                linewidth=.5,
                c='lime',
                edgecolors='black',
            )
            if 'scatternum' == annotate:
                for i, x_col in enumerate(x_cols):
                    ax.annotate(
                        f'{i}',
                        (series[x_col], series[y_cols[i]]),
                        fontsize=6,
                    )
    plt.title(title)
    plt.tight_layout()
    if save_fig:
        path = f'./figs/images/{category}'
        if not Path(path).exists():
            os.makedirs(path)
        # NOTE: File names would look nicer if we remove the original
        #       suffix here, but it will be easier to tie back to the
        #       original image if we don't.
        if annotate:
            filename += f'_{annotate}'
        plt.savefig(
            f'{path}/{filename}.png',
            dpi=300,
            bbox_inches='tight'
        )
    plt.show()

#-- scrape all images
# for i in range(df.shape[0]):
#     data.get_image(i)

#-- try a few plots
for i in [4]:#range(10):
    for annotate in [None,'scatter','scatternum','spline','splinelabel']:
        plot_image(i,annotate=annotate,save_fig=False)


print(f'target counts:\n{df.loc[:,target_cols].sum()}\n')

no_targets = reduce(iand, [df[col] == 0 for col in target_cols])
print(f'no targets:  {df[no_targets].shape}')
print(f'one or more: {df[~no_targets].shape}')
