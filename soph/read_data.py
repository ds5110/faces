#Sophia Cofone 11/19 
#File is intended to read df and select some general groupings of features

import pandas as pd
import numpy as np

#gets the whole dataset
def get_data(data='data/merged_landmarks.csv'):    
    df = pd.read_csv(data,dtype={
        'image-set': str,
        'filename': str,
        'partition': str,
        'subpartition': str,})
    return df

'''
Defining some groups of features (see the derived.md for more information on the features)
Output should be a list of the columns so the helper (get_Xy) can appropriatly split the data
'''
def get_categories(df):
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
    '''a minimal set of coords with one xy point for: 
    leye,reye
    lbrow,rbrow
    lear,rear
    lcheek,rcheek
    mouth
    nose
    chin
    '''
    selected_d_coords = [0,16,19,24,37,44,29,3,13,8,66]
    f_num_list = np.sort(selected_d_coords)
    selected_d_coords = []
    for i in f_num_list:
        for j in f_num_list:
            if ('dist_{}_{}'.format(i,j) not in selected_d_coords) and (('dist_{}_{}'.format(j,i) not in selected_d_coords)) and i!=j:
                selected_d_coords.append('dist_{}_{}'.format(i,j))

    return orig_coords, angle_off, main_predictors,norm_cenrot_sym_diff, norm_cenrot,all_d_coords, selected_d_coords