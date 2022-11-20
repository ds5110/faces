#Sophia Cofone 11/19 
#File is intended to read df and select some general groupings of features

import pandas as pd

#gets the whole dataset
def get_data():    
    df = pd.read_csv('data/merged_landmarks.csv',dtype={
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
    
    # #Derived Landmark Coordinates
    # #normalized to the minimum bounding box of landmarks
    # df1 = df.loc[:, df.columns.str.startswith('cenrot_sym_diff')]
    # norm = list(df1.columns)
    # #transformed to correct yaw and position
    # cenrot = 
    # #transformed to correct yaw and position
    # #normalized to the new minimum bounding box
    # norm_cenrot =

    #Differences of expected symmetric landmarks
    #Difference of yaw-corrected coordinates
    df1 = df.loc[:, df.columns.str.startswith('cenrot_sym_diff')]
    cenrot_sym_diff = list(df1.columns)
    #Difference of yaw-corrected coordinates, normalized per min box
    df1 = df.loc[:, df.columns.str.startswith('norm_cenrot_sym_diff')]
    norm_cenrot_sym_diff = list(df1.columns)
    
    return orig_coords, angle_off, main_predictors,cenrot_sym_diff,norm_cenrot_sym_diff

def main():
    df = get_data()
    orig_coords, angle_off, main_predictors,cenrot_sym_diff,norm_cenrot_sym_diff = get_categories(df)

if __name__ == "__main__":
    main()