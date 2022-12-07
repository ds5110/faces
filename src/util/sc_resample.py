#Sophia Cofone 11/18/22
'''
File is intended to house functions used for up and downsampling.
We will use logistic regression to see the differences in the sampling methods (for example).

reference https://elitedatascience.com/imbalanced-classes
'''

#basic
import pandas as pd
#sklearn
from sklearn.utils import resample

def split_df(df):
    df_baby = df[df['baby']==1]
    df_adult = df[df['baby']==0]

    return df_baby, df_adult

def upsample(df):
    df_baby, df_adult = split_df(df)
    df_minority_upsampled = resample(df_baby, 
    replace=True,
    n_samples=689,#adult class
    random_state=42)
    #Combine adult with upsampled baby
    df_upsampled = pd.concat([df_adult, df_minority_upsampled])

    return df_upsampled

def downsample(df):
    df_baby, df_adult = split_df(df)
    df_majority_downsampled = resample(df_adult, 
    replace=False,
    n_samples=410,#baby class
    random_state=42) 
    #Combine baby with downsampled adult
    df_downsampled = pd.concat([df_majority_downsampled, df_baby])

    return df_downsampled