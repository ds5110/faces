#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:51:12 2022

row 3 'left_eye_top' is problematic for 3-spline

@author: jhautala
"""

from operator import iand
from functools import reduce

# intra-project
from util.local_cache import cache
from util.model import cat_cols

class ImagePlus:
    def __init__(self,img,model):
        self.img = img
        self.model = model

# load the labels data
df = cache.get_meta()

def scrape_all():
    for i in range(df.shape[0]):
        cache.get_image(i)

def check_categories():
    no_cats = reduce(iand, [df[col] == 0 for col in cat_cols])
    print(f'no categories:  {df[no_cats].shape}')
    print(f'one or more:    {df[~no_cats].shape}')

def get_image_by_name():
    cache.get_image(path='ads',file='1072.Still005.jpg')
    
if __name__ == '__main__':
    # scrape_all()
    check_categories()
