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

pi_2 = np.pi/2.
to_deg = 180./np.pi
to_rad = np.pi/180.
id_2d = np.array([[1,0],[0,1]])

class Sym:
    def __init__(self,desc,pairs,weight=1.):
        self.desc = desc
        self.pairs = np.array(pairs)
        self.weight = weight
        self.weight_tot = len(self.pairs) * weight
    def get_left(self):
        return self.pairs[:,0]
    def get_right(self):
        return self.pairs[:,1]

syms = [
    # Sym(
    #     'cheeks',
    #     ((i, 16 - i) for i in range(8)),
    #     .5,
    # ),
    # just the corners of the eyes seem like a good-enough way to normalize tilt
    Sym(
        'eyes_corners',
        [
            [36, 45],
            [39, 42],
        ],
        4. # higher weight for fewer points
    ),
    Sym(
        'eyes',
        [
            # [36, 45],
            [37, 44],
            [38, 43],
            # [39, 42],
            [40, 47],
            [41, 46],
        ],
        .5, # due to squinting/expressions and number of points
    ),
]

def __get_angle(anno,sym):
    # calculate diffs per pair
    left, right = np.squeeze(np.split(anno.get_coords()[sym.pairs.T],2))
    xx, yy = np.squeeze(np.split((right-left).T,2))
    hypots = np.sqrt(xx**2 + yy**2)
    weight_tot = np.sum(hypots)
    
    # calculate angle
    angles = np.arcsin(-yy/hypots) # neg y because image y starts at top
    angle = np.sum(angles * hypots)/weight_tot
    
    # check for upside down (not likely, eh)
    angles = np.arccos(xx/hypots)
    upside_down = np.sign(np.sum(angles * hypots)) == -1
    if upside_down:
        angle = np.pi - angle
    
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
    
    # add 'rotated' to the image description
    desc = ' '.join(d for d in [anno.desc, 'rotated'] if d is not None)
    return AnnoImg(
        anno.image_set,
        anno.filename,
        coords,
        _img,
        desc=desc
    )
