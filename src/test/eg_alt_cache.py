#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:19:12 2022

@author: jhautala
"""

# project
from util import cache, alt
from util.pre import get_yaw_data
from main.decorate import get_decorated_meta

df = cache.get_meta()

# use the alternate image sets
df2 = alt.get_meta()

# grab an example image
anno = alt.get_image(671)

# directly grab rotation data
_, _, face, angle, coords = get_yaw_data(anno)

# add a bunch of derived columns
df2 = get_decorated_meta(alt)
alt.save_meta(df2, 'decorated')

# reload from disk
df2 = alt.get_meta('decorated')
