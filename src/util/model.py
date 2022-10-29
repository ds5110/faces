#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:58:49 2022

@author: jhautala
"""

import numpy as np

cat_cols = ['turned', 'occluded', 'tilted', 'expressive']
x_cols, y_cols = [[f'gt-{axis}{i}' for i in range(68)] for axis in ['x','y']]

class Feature:
    def __init__(self, desc, idx):
        self.desc = desc
        self.idx = idx

class LandmarkModel:
    def __init__(self,features):
        self.features = features
        self.per_desc = {f.desc for f in features}

nose_i = 33
landmark68 = LandmarkModel(
    [
        Feature('right_cheek', range(8)), # overlaps chin by 1
        Feature('chin', range(7,10)), # center at 8
        Feature('left_cheek', range(9,17)), # overlaps chin by 1
        Feature('right_brow', range(17,22)),
        Feature('left_brow', range(22,27)),
        Feature('nose_v', range(27,31)),
        Feature('nose_h', range(31,36)),
        Feature('right_eye_top', range(36,40)), # corners at 36 and 39
        Feature('right_eye_bot', [*range(39,42),36]), # corners at 36 and 39
        Feature('left_eye_top', range(42,46)), # corners at 42 and 45
        Feature('left_eye_bot', [*range(45,48),42]), # corners at 42 and 45
        Feature('lip_out_top_right', range(48,51)), # overlaps philtrum by 1
        Feature('philtrum', range(50,53)), # center at 51
        Feature('lip_out_top_left', range(52,55)), # overlaps philtrum by 1
        Feature('lip_out_bot', range(55,60)),
        Feature('lip_in_top', range(60,65)),
        Feature('lip_in_bot', [*range(64,68),60]),
    ]
)

class AnnoImg:
    def __init__(
            self,
            image_set,
            filename,
            coords,
            im_fun,
            row_id=None,
            desc=None,
    ):
        self.image_set = image_set
        self.filename = filename
        self.coords = coords
        self.im_fun = im_fun
        self.face = np.stack([[vv[nose_i] for vv in [self.get_x(), self.get_y()]]])
        self.row_id = row_id
        self.desc = desc
    
    def get_image(self):
        return self.im_fun()

    def get_x(self):
        return self.coords[0]
    
    def get_y(self):
        return self.coords[1]
