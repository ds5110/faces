#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:41:53 2022

@author: jhautala
"""

import numpy as np

to_deg = 180./np.pi
to_rad = np.pi/180.

class Sym:
    def __init__(self,desc,ii,weight=1.):
        self.desc = desc
        self.ii = dict(ii)
        self.weight = weight
        self.weight_tot = len(self.ii) * self.weight
    
    def pairs(self):
        return self.ii.items()

syms = [
    # Sym('cheeks', ((i, 16 - i) for i in range(8))),
    # just the corners of the eyes seem like a good way to normalize tilt
    Sym(
        'eyes',
        (
            (36, 45),
            (39, 42),
        ),
        4.
    ),
    # Sym(
    #     'eyes',
    #     (
    #         (36, 45),
    #         (37, 44),
    #         (38, 43),
    #         (39, 42),
    #         (40, 47),
    #         (41, 46),
    #     ),
    #     2.
    # ),
]

def __get_angle(df,row_id,sym):
    xx_left = []
    yy_left = []
    xx_right = []
    yy_right = []
    for i_left, i_right in sym.pairs():
        xx_left.append(df[f'gt-x{i_left}'][row_id])
        yy_left.append(df[f'gt-y{i_left}'][row_id])
        xx_right.append(df[f'gt-x{i_right}'][row_id])
        yy_right.append(df[f'gt-y{i_right}'][row_id])
    xx = np.array(xx_right) - np.array(xx_left)
    yy = np.array(yy_left) - np.array(yy_right)
    hypots = np.sqrt(xx**2 + yy**2)
    angles = np.arcsin(yy/hypots)
    angle = np.sum(angles * hypots)/np.sum(hypots)
    return angle, sym.weight_tot

def get_angle(df,row_id,deg=False):
    # NOTE: left and right here refer to orientation in the image, not anatomy
    angles = []
    weights = []
    for sym in syms:
        angle, weight = __get_angle(df,row_id,sym)
        angles.append(angle)
        weights.append(weight)
    angles = np.array(angles)
    weights = np.array(weights)
    angle = np.sum(angles * weights) / np.sum(weights)
    if deg:
        return to_deg * angle
    else:
        return angle