#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:41:53 2022

@author: jhautala
"""

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.interpolate import UnivariateSpline
import skimage

# intra-project
from util.model import AnnoImg, landmark68

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

cheeks = np.array([[i,16-i] for i in range(8)])
brows = np.array([[17+i, 26-i] for i in range(5)])
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
    *cheeks,
    
    # brows
    *brows,
    
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
        4. # higher weight for canthi
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

def _get_yaw(coords,sym):
    '''
    This function calculates the angle offset based on the given
    coordinates and expected symmetric point pairs.

    Parameters
    ----------
    coords : number with shape(landmarks,dimensions)
        the landmark coordinates to check.
    sym : Sym
        The basis of symmetry (essentially a combination of point pairs).

    Returns
    -------
    angle : float
        The estimated angle of rotation (in radians).

    '''
    # calculate diffs per pair
    left, right = np.squeeze(np.split(coords[sym.pairs.T],2))
    xx, yy = np.squeeze(np.split((right-left).T,2))
    yy = -yy # neg y because image y starts at top
    
    # calculate pair distances
    hypots = np.sqrt(xx**2 + yy**2)
    weight_tot = np.sum(hypots) # use pair distance as weight
    
    x = np.sum(xx * hypots)/weight_tot
    y = np.sum(yy * hypots)/weight_tot
    
    # calculate angle
    y_neg = np.sign(y) == -1
    if x == 0:
        angle = -pi_2 if y_neg else pi_2
    else:
        angle = np.arctan(y/x)
        x_neg = np.sign(x) == -1
        if x_neg:
            if y_neg:
                angle -= np.pi
            else:
                angle += np.pi
    
    return angle

def get_yaw(anno,syms=default_syms,deg=False):
    coords = anno.get_coords()
    angles = []
    weights = []
    for sym in syms:
        angle = _get_yaw(coords,sym)
        angles.append(angle)
        weights.append(sym.weight_tot)
    angles = np.array(angles)
    weights = np.array(weights)
    angle = np.sum(angles * weights) / np.sum(weights)
    if deg:
        return to_deg * angle
    else:
        return angle

def get_yaw_data(anno):
    '''
    Extract image metadata associated with center/rotate logic.

    Parameters
    ----------
    anno : AnnoImg
        the image to analyze.

    Returns
    -------
    in_dims 2-item array of number
        original image dimensions.
    margin 2-item array of number
        margin size to fit features such that in_dims + 2*margin = out_dims.
    face : array of number
        original coordinates of the center of the face.
    angle : number
        estimated angle of rotation (in radians).
    coords : number(68,2)
        new landmark coordinates, rotated and centered.

    '''
    # anno = cache.get_image(118)

    # get image and calculate translation
    img = anno.get_image()
    face = anno.get_face_center()
    center = [np.nan, np.nan] if img is None else np.array([[img.width/2, img.height/2]])
    coords = anno.get_coords()
    
    # calculate rotation
    angle = get_yaw(anno)
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotx = np.array([[cos,sin],[-sin,cos]])
    
    # perform rotation
    coords = coords - face # move coordinates to the origin to rotate
    coords = coords@rotx # apply rotation matrix
    if img is not None:
        coords = coords + center # move coordinates to image center
    else:
        # NOTE: This is currently only here support the case where raw image data is unavailable...
        #       Unfortunately, it complicates user expectations for this function
        coords = coords + face # move coordinates back to original location
    
    # calculate buffer
    mins = np.amin(coords,axis=0)
    maxs = np.amax(coords,axis=0)
    if img is None:
        in_dims = np.array([np.nan, np.nan])
        margin = np.array([np.nan, np.nan])
    else:
        in_dims = np.array([img.width, img.height])
        margin = []
        for i in range(2):
            max_buff = 0
            lo_buff = 0 - mins[i]
            if lo_buff > max_buff:
                max_buff = lo_buff
            hi_buff = maxs[i] - in_dims[i]
            if hi_buff > max_buff:
                max_buff = hi_buff
            margin.append(max_buff + max_buff if max_buff > 0 else 0)

        margin = np.array(margin)
        coords += margin
    return in_dims, margin, face, angle, coords

def rotate(anno):
    def _img():
        img = anno.get_image()
        in_dims, margin, face, angle, coords = get_yaw_data(anno)
        
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
        crop = rot.crop(
            (
                img.width - margin[0],
                img.height - margin[1],
                img.width*2 + margin[0],
                img.height*2 + margin[1]
            )
        )
        
        return crop
    
    _, _, _, _, coords = get_yaw_data(anno)
    
    # add 'rotated' to the image description
    desc = anno.desc.copy().append('rotated')
    return AnnoImg(
        anno.image_set,
        anno.filename,
        coords,
        _img,
        anno.row_id,
        desc=desc
    )

def crop(anno,use_splines=False):
    def _img():
        image = np.array(anno.get_image()) # convert to skimage
        coords = anno.get_coords()
        if use_splines:
            X = []
            Y = []
            for f in landmark68.features:
                # f = landmark68.features[0]
                data = coords[f.idx]
                if np.any(np.isnan(data)): continue;
                points = data.T
                [xx, yy] = points
                distance = np.cumsum(
                    np.sqrt(np.sum(
                        np.diff(points.T, axis=0)**2,
                        axis=1
                    ))
                )
                if not distance[-1]: continue;
                
                distance = np.insert(distance, 0, 0)/distance[-1]
                splines = [UnivariateSpline(distance, point, k=2, s=.2) for point in points]
                points_fitted = np.vstack(
                    [spline(np.linspace(0, 1, 64)) for spline in splines]
                )
                points_fitted.shape
                X.extend(points_fitted[0])
                Y.extend(points_fitted[1])
            
            data = np.stack([X,Y],axis=1)
        else:
            data = coords
        data = data[~np.isnan(data).any(axis=1),:]
        poly = data[ConvexHull(data).vertices]
        X, Y = skimage.draw.polygon(poly[:,0], poly[:,1])
        cropped = np.zeros(image.shape, dtype=np.uint8)
        cropped[Y, X] = image[Y, X]
        return Image.fromarray(cropped)
    
    desc = anno.desc.copy().append('cropped')
    return AnnoImg(
        anno.image_set,
        anno.filename,
        anno.get_coords(),
        _img,
        anno.row_id,
        desc=desc
    )
