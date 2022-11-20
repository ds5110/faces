#

from read_data import get_data

import pandas as pd
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

def main():
    df = get_data()
    explore_shape(df)

if __name__ == "__main__":
    main()


    