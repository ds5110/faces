#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:09:16 2022

@author: jhautala
"""

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
from PIL import Image

# intra-project
from util.local_cache import LocalCache
from util.pre import get_angle, to_deg
from util.model import x_cols, y_cols

id_2d = np.array([[1,0],[0,1]])
nose_i = 27

cache = LocalCache()

df = cache.get_meta()

for row_id in [1]:#range(10):
    # row_id = 4
    category = df['image-set'][row_id]
    filename = df['filename'][row_id]
    title = f'{category}/{filename}'
    if row_id is not None:
        title += f' (row {row_id})'
    
    # extract data
    coords = np.stack([df[cols].loc[row_id,:].values for cols in [x_cols, y_cols]])
    xx = coords[0]
    yy = coords[1]
    face = np.stack([[vv[nose_i] for vv in [xx, yy]]])
    
    # get image and calculate translation
    img = cache.get_image(row_id)
    center = np.array([[img.width/2, img.height/2]])
    cen = img.transform(
        img.size,
        method=Image.Transform.AFFINE,
        data=np.append(id_2d, (face-center).T, 1).ravel(),
    )
    
    # calculate rotation
    coords -= face.T # move coordinates to the origin to rotate
    angle = get_angle(df,row_id)
    angle_deg = angle * to_deg
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotx = np.array([[cos,sin],[-sin,cos]])
    # coords = rotx@coords
    coords = (coords.T@rotx).T
    coords += center.T # move coordinates back to the center
    
    # def doit(desc,tmp,cross=False):
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.imshow(tmp)
    #     if cross:
    #         ax.axhline(y=center[0,1])
    #         ax.axvline(x=center[0,0])
    #         # ax.axhline(y=y_face)
    #         # ax.axvline(x=x_face)
    #     # plt.xlim()
    #     plt.title(title + f' {desc}')
    #     plt.tight_layout()
    #     plt.show()
    
    # doit('original',img,True)
    
    
    # doit('centered',cen,True)
    
    rot = cen.rotate(
        -angle_deg,
        # expand=True, # doesn't work with simple centering
        center=(center[0,0],center[0,1]),
        # translate=(
        #     x_center-x_face,
        #     y_center-y_face
        # ),
    )
    # doit('rotated',rot,True)
    
    # a = np.cos(angle)
    # b = np.sin(angle)
    # c = x_center - x_center * a - y_center * b
    # d = -b
    # e = a
    # f = y_center - x_center * d - y_center * e
    
    # cenx = np.array([
    #     [1, 0, x_center-x_face],
    #     [0, 1, y_center-y_face,],
    # ])
    
    # face@cenx
    # bothx = np.array([[a,b,c],[d,e,f]])
    # onlyrot = np.array([[a,b,0],[d,e,0]])
    # rotx = np.array([[a,b],[d,e]])
    # def test_mult(desc,p):
    #     print(f'{desc}: {p}')
    #     print(f'rotx@{desc}: {rotx@p}')
    #     print(f'{desc}@rotx: {p@rotx}')
    # test_mult('100', np.array([100,100]))
    # test_mult('center',center)
    # test_mult('face',face)
    
    # both = img.transform(
    #     img.size,
    #     method=Image.Transform.AFFINE,
    #     data=onlyrot.ravel(),
    # )
    # doit('both',both,True)
    
    # title += ' rotated'
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rot)
    ax.scatter(
        coords[0],
        coords[1],
        s=6,
        linewidth=.5,
        c='lime',
        edgecolors='black',
    )
    ax.axhline(y=center[0,1])
    ax.axvline(x=center[0,0])
    # ax.axhline(y=y_face)
    # ax.axvline(x=x_face)
    # plt.xlim()
    plt.title(title)
    plt.tight_layout()
    plt.show()