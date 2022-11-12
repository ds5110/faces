#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:41:53 2022

@author: jhautala
"""

import numpy as np
from PIL import Image

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

h_syms = np.array([
    [36, 45], # outer canthus
    [39, 42], # inner canthus
    
    # eyelids
    [37, 44],
    [38, 43],
    [40, 47],
    [41, 46],
    
    # h_nose
    [31, 35],
    [32, 34],
    
    # cheeks
    *[[i,16-i] for i in range(8)],
    
    # brows
    *[[17+i, 26-i] for i in range(5)],
    
    # mouth
    *[[48+i, 54-i] for i in range(3)],
    *[[60+i, 64-i] for i in range(2)],
    [67, 65],
    *[[59-i, 55+i] for i in range(2)],
])

v_line = np.array([
    *[28+i for i in range(4)],
    34,
    52,
    63,
    67,
    58,
    9,
])



default_syms = [
    # NOTE: I haven't tested this a lot, but I don't expect
    #       it will be a great/reliable way to address tilt...
    # Sym(
    #     'cheeks',
    #     ((i, 16 - i) for i in range(8)),
    #     .5,
    # ),
    # NOTE: The corners of the eyes seem good-enough on their own
    #       for the purpose of normalizing yaw
    Sym(
        'canthi',
        [
            [36, 45],
            [39, 42],
        ],
        2. # higher weight for canthi
    ),
    # NOTE: This is probably not worth keeping, especially
    #       since its weight is so low...
    Sym(
        'eyelids',
        [
            # [36, 45], # already included in eyes corners
            [37, 44],
            [38, 43],
            # [39, 42], # already included in eyes corners
            [40, 47],
            [41, 46],
        ],
        .5, # due to expressions (e.g. squinting)
    ),
]

def _get_angle(anno,sym):
    '''
    This function calculates the angle offset based on the given
    image and expected symmetric point pairs.

    Parameters
    ----------
    anno : AnnoImg
        The image to check.
    sym : Sym
        The basis of symmetry (essentially a combination of point pairs).

    Returns
    -------
    angle : float
        The estimated angle of rotation (in radians).

    '''
    # calculate diffs per pair
    # NOTE: left and right here refer to viewer's perspective
    left, right = np.squeeze(np.split(anno.get_coords()[sym.pairs.T],2))
    xx, yy = np.squeeze(np.split((right-left).T,2))
    
    # calculate pair distances
    hypots = np.sqrt(xx**2 + yy**2)
    weight_tot = np.sum(hypots) # use pair distance as weight
    
    # calculate angle
    angles = np.arcsin(-yy/hypots) # neg y because image y starts at top
    angle = np.sum(angles * hypots)/weight_tot
    
    # check for upside down (not likely, eh)
    angles = np.arccos(xx/hypots)
    upside_down = np.sign(np.sum(angles * hypots)) == -1
    if upside_down:
        angle = np.pi - angle
    
    return angle

def get_angle(anno,syms=default_syms,deg=False):
    angles = []
    weights = []
    for sym in syms:
        angle = _get_angle(anno,sym)
        angles.append(angle)
        weights.append(sym.weight_tot)
    angles = np.array(angles)
    weights = np.array(weights)
    angle = np.sum(angles * weights) / np.sum(weights)
    if deg:
        return to_deg * angle
    else:
        return angle

def get_rotate_data(anno):
    '''
    Extract image metadata associated with center/rotate logic.

    Parameters
    ----------
    anno : AnnoImg
        the image to analyze.

    Returns
    -------
    width number
        image width.
    height number
        image height.
    face : array of number
        coordinates of the center of the face.
    angle : number
        estimated angle of rotation (in radians).
    coords : number(68,2)
        new landmark coordinates, rotated and centered.

    '''
    # get image and calculate translation
    img = anno.get_image()
    face = anno.get_face_center()
    center = np.array([[img.width/2, img.height/2]])
    coords = anno.get_coords()
    
    # calculate rotation
    angle = get_angle(anno)
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotx = np.array([[cos,sin],[-sin,cos]])
    
    # perform rotation
    coords = coords - face # move coordinates to the origin to rotate
    coords = coords@rotx # apply roatation matrix
    coords = coords + center # move coordinates back to the center
    
    return img.width, img.height, face, angle, coords

def rotate(anno):
    def _img():
        img = anno.get_image()
        _, _, face, angle, coords = get_rotate_data(anno)
        
        # NOTE: We add a buffer around the image
        #       to avoid cropping content during centering
        buff = Image.new(img.mode, (img.width*3, img.height*3))
        buff.paste(img, (img.width, img.height))
        buff_face = face + np.array([[img.width,img.height]])
        buff_cen = np.array([[buff.width/2, buff.height/2]])
        cen = buff.transform(
            buff.size,
            method=Image.Transform.AFFINE,
            data=np.append(id_2d, (buff_face-buff_cen).T, 1).ravel(),
        )
        
        angle_deg = angle * to_deg
        rot = cen.rotate(
            -angle_deg,
            center=tuple(buff_cen.ravel()),
        )
        
        # crop back to original size
        crop = rot.crop((img.width,img.height,img.width*2,img.height*2))
        
        return crop
    
    _, _, _, _, coords = get_rotate_data(anno)
    
    # add 'rotated' to the image description
    desc = ' '.join(d for d in [anno.desc, '(rotated)'] if d is not None)
    return AnnoImg(
        anno.image_set,
        anno.filename,
        coords,
        _img,
        desc=desc
    )
