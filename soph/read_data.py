#Sophia Cofone 11/19 
'''
This file is primarily intended for reading the data frame.
There is also a function for generating lists of column names for general "grouings" of features. This makes it a little easier to test different features.
'''

#basic
import pandas as pd
import numpy as np

def get_data(data='data/merged_landmarks.csv'):    
    df = pd.read_csv(data,dtype={
        'image-set': str,
        'filename': str,
        'partition': str,
        'subpartition': str,})
    return df


def get_categories(df):
    '''
    Defining some groups of features (see the derived.md for more information on the features)
    Output should be a list of the columns so the helper function (get_Xy) can appropriatly split the data
    '''
    #Original Landmark Coordinates
    coord_nums = range(68)
    orig_coords = ['{}{}'.format(dim,i) for i in coord_nums for dim in ['x','y']]

    #Angular offsets
    '''
    yaw: estimated rotation about the z-axis
    yaw_abs: magnitude of yaw
    roll: estimated rotation about the y-axis
    roll_abs: magintude of roll
    '''
    angle_off = ['yaw', 'yaw_abs','roll','roll_abs']

    #Potential predictors for distinguishing infants
    main_predictors = ['boxratio', 'interoc','interoc_norm','boxsize','boxsize/interoc']
    
    #Derived Landmark Coordinates
    #transformed to correct yaw and position & normalized to the new minimum bounding box
    df1 = df.loc[:, df.columns.str.startswith('norm_cenrot-')]
    norm_cenrot = list(df1.columns)

    #Difference of expected symmetric landmarks, yaw-corrected coordinates, normalized per min box
    df1 = df.loc[:, df.columns.str.startswith('norm_cenrot_sym_diff')]
    norm_cenrot_sym_diff = list(df1.columns)

    #Euc distance of all coords (yaw and box corrected)
    #all of them
    df1 = df.loc[:, df.columns.str.startswith('dist_')]
    all_d_coords = list(df1.columns)

    return orig_coords, angle_off, main_predictors,norm_cenrot_sym_diff, norm_cenrot,all_d_coords