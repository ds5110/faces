#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 16:58:49 2022

@author: jhautala
"""

import numpy as np


# ----- landmark indices
nose_i = 33  # the center of the horizontal line under nose
cheeks = np.array([[i, 16 - i] for i in range(8)])
brows = np.array([[17 + i, 26 - i] for i in range(5)])
h_syms = np.array([
    [36, 45],  # outer canthi
    [39, 42],  # inner canthi

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
    *[[48 + i, 54 - i] for i in range(3)],
    *[[60 + i, 64 - i] for i in range(2)],
    [67, 65],
    *[[59 - i, 55 + i] for i in range(2)],
])

# points that should be centered in frontal view
v_line = np.array([
    *[28 + i for i in range(4)],
    34,
    52,
    63,
    67,
    58,
    9,
])


# TODO: reconcile with column_names.py
'''
Defining some groups of features (see the derived.md for more information on the features)
Output should be a list of the columns so the helper function (get_Xy) can appropriatly split the data
'''
cat_cols = ['turned', 'occluded', 'tilted', 'expressive']
x_cols, y_cols = [[f'gt-{axis}{i}' for i in range(68)] for axis in ['x','y']]
landmark_cols = [x_cols, y_cols]
cenrot_cols = [[f'cenrot-{axis}{i}' for i in range(68)] for axis in ['x','y']]
norm_cols = [[f'norm_cenrot-{axis}{i}' for i in range(68)] for axis in ['x','y']]
nose_i = 33  # the center of the horizontal line under nose
outer_canthi = [36, 45]
#Potential predictors for distinguishing infants
main_predictors = ['boxratio', 'interoc','interoc_norm','boxsize','boxsize/interoc']
#Angular offsets
angle_off = ['yaw', 'yaw_abs','roll','roll_abs']


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


# NOTE: This model provides access to image data through an injected
#       supplier function to reduce its memory cost.
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
        
        self.im_fun = im_fun
        self.row_id = row_id
        self.desc = desc if desc is not None else []
        
        self.set_coords(coords)
    
    def get_image(self):
        return self.im_fun()
    
    def get_coords(self):
        return np.copy(self._coords)
    
    def set_coords(self, coords):
        self._coords = coords
        
        # - extract face center as center of nose_h
        xx, yy = [coords[:,i] for i in range(2)]
        nose_x, nose_y = [vv[nose_i] for vv in [xx, yy]]
        self._face = np.stack([nose_x, nose_y])
    
    def get_face_center(self):
        return np.copy(self._face)

    def get_x(self):
        return np.copy(self._coords[:,0])
    
    def get_y(self):
        return np.copy(self._coords[:,1])
