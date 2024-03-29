#Sophia Cofone 11/19 
'''
This file is intended to capture the EDA done for the logreg model.
'''
#project
from util.sc_helpers import get_data
#basic
import seaborn as sns
import matplotlib.pyplot as plt

def explore_shape(df):
    df_baby = df[df['baby']==1]
    print(f"Shape of baby df: {df_baby.shape}")
    df_adult = df[df['baby']==0]
    print(f"Shape of adult df: {df_adult.shape}")
    num_adult_points = df_adult.shape[0]
    num_baby_points = df_baby.shape[0]
    total_points = num_adult_points + num_baby_points
    adult_p = round(num_adult_points/total_points,2)*100
    baby_p = round(num_baby_points/total_points,2)*100
    print(f"Total dataset is {adult_p}% adult points and {baby_p}% baby points")
    sns.countplot(data=df, x='baby')
    plt.show()

    return df_baby, df_adult

def histo_main_predictors(df_predictors):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Comparing distrobution of "potential predictors"')
    sns.histplot(ax=axes[0, 0], data=df_predictors, x='boxratio',hue='baby')
    sns.histplot(ax=axes[0, 1], data=df_predictors, x='interoc',hue='baby')
    sns.histplot(ax=axes[0, 2], data=df_predictors, x='interoc_norm', hue='baby')
    sns.histplot(ax=axes[1, 0], data=df_predictors, x='boxsize', hue='baby')
    sns.histplot(ax=axes[1, 1], data=df_predictors, x='boxsize/interoc', hue='baby')
    plt.show()

def histo_angle_off(df_predictors):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle('Comparing distrobution of "angular offsets"')
    sns.histplot(ax=axes[0, 0], data=df_predictors, x='yaw',hue='baby')
    sns.histplot(ax=axes[0, 1], data=df_predictors, x='yaw_abs',hue='baby')
    sns.histplot(ax=axes[0, 2], data=df_predictors, x='roll', hue='baby')
    sns.histplot(ax=axes[1, 0], data=df_predictors, x='roll_abs', hue='baby')
    plt.show()

def main():
    df = get_data()
    explore_shape(df)
    histo_main_predictors(df)
    histo_angle_off(df)

if __name__ == "__main__":
    main()