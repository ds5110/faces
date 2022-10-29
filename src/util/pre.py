#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:41:53 2022

@author: jhautala
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# intra-project
from util.model import AnnoImg

to_deg = 180./np.pi
to_rad = np.pi/180.
id_2d = np.array([[1,0],[0,1]])

class Sym:
    def __init__(self,desc,ii,weight=1.):
        self.desc = desc
        ii = np.array(ii)
        self.weight = weight
        self.weight_tot = len(ii) * weight
        self.pairs = ii.T
    def get_left(self):
        return self.pairs.T[0]
    def get_right(self):
        return self.pairs.T[1]

syms = [
    # Sym('cheeks', ((i, 16 - i) for i in range(8))),
    # just the corners of the eyes seem like a good way to normalize tilt
    Sym(
        'eyes',
        [
            [36, 45],
            [39, 42],
        ],
        4.
    ),
    # Sym(
    #     'eyes',
    #     [
    #         [36, 45],
    #         [37, 44],
    #         [38, 43],
    #         [39, 42],
    #         [40, 47],
    #         [41, 46],
    #     ],
    #     2.
    # ),
]

def __get_angle(anno,sym):
    xx_left = anno.get_x()[sym.pairs[0]]
    xx_right = anno.get_x()[sym.pairs[1]]
    yy_left = anno.get_y()[sym.pairs[0]]
    yy_right = anno.get_y()[sym.pairs[1]]
    xx = np.array(xx_right) - np.array(xx_left)
    yy = np.array(yy_left) - np.array(yy_right)
    hypots = np.sqrt(xx**2 + yy**2)
    angles = np.arcsin(yy/hypots)
    angle = np.sum(angles * hypots)/np.sum(hypots)
    return angle, sym.weight_tot

def get_angle(anno,deg=False):
    # NOTE: left and right here refer to orientation in the image, not anatomy
    angles = []
    weights = []
    for sym in syms:
        angle, weight = __get_angle(anno,sym)
        angles.append(angle)
        weights.append(weight)
    angles = np.array(angles)
    weights = np.array(weights)
    angle = np.sum(angles * weights) / np.sum(weights)
    if deg:
        return to_deg * angle
    else:
        return angle

def rotate(anno):
    # extract data
    
    # get image and calculate translation
    def _rot():
        img = anno.get_image()
        face_cen = anno.get_face_center()
        center = np.array([[img.width/2, img.height/2]])
        coords = anno.get_coords()
        
        # TODO handle cropping
        buff = Image.new(img.mode, (img.width*3, img.height*3))
        buff.paste(img, (img.width, img.height))
        buff_face = face_cen + np.array([[img.width,img.height]])
        buff_cen = np.array([[buff.width/2, buff.height/2]])
        cen = buff.transform(
            buff.size,
            method=Image.Transform.AFFINE,
            data=np.append(id_2d, (buff_face-buff_cen).T, 1).ravel(),
        )
        
        # calculate rotation
        coords = coords - face_cen # move coordinates to the origin to rotate
        angle = get_angle(anno)
        angle_deg = angle * to_deg
        cos = np.cos(angle)
        sin = np.sin(angle)
        rotx = np.array([[cos,sin],[-sin,cos]])
        coords = coords@rotx
        coords = coords + center # move coordinates back to the center
        
        rot = cen.rotate(
            -angle_deg,
            center=tuple(buff_cen.ravel()),
        )
        
        # crop back to original size
        crop = rot.crop((img.width,img.height,img.width*2,img.height*2))
        
        return crop, coords
    def _img():
        rot, coords = _rot()
        return rot
    rot, coords = _rot()
    
    return AnnoImg(
        anno.image_set,
        anno.filename,
        coords,
        _img,
        desc='rotated'
    )
