#Sophia Cofone 11/20/22
'''
This file is intended to capture the EDA done for the "distance" idea.
'''

#project
from util.sc_helpers import get_data
#basic
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#sklearn
from sklearn.metrics.pairwise import euclidean_distances


def matrix_one(df):
    #getting the coordinate pairs for one image (into list)
    anot_data = df.iloc[0]
    anot_coords = []
    for coord in range(68):
        x = anot_data['norm-x{}'.format(coord)]
        y = anot_data['norm-y{}'.format(coord)]
        anot_coords.append(np.array([x,y]))
    #getting the difference between coordinate pairs for one image
    dist_mat = euclidean_distances(anot_coords)
    plt.imshow(dist_mat)
    plt.show()

def cor_mat(df):
    df_dist = df.loc[:, df.columns.str.startswith('dist_')]
    matrix = df_dist.corr().abs()
    plt.imshow(matrix)
    plt.show()

def histo_predictors(df_predictors):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8))
    fig.suptitle('Comparing distrobution of "Euclidian distance and boxratio"')
    sns.histplot(ax=axes[0, 0], data=df_predictors, x='boxratio',hue='baby')
    sns.histplot(ax=axes[0, 1], data=df_predictors, x='dist_5_7',hue='baby')
    sns.histplot(ax=axes[0, 2], data=df_predictors, x='dist_7_9', hue='baby')
    sns.histplot(ax=axes[0, 3], data=df_predictors, x='dist_7_48', hue='baby')
    sns.histplot(ax=axes[1, 0], data=df_predictors, x='dist_18_25', hue='baby')
    sns.histplot(ax=axes[1, 1], data=df_predictors, x='dist_21_22', hue='baby')
    sns.histplot(ax=axes[1, 2], data=df_predictors, x='dist_33_42', hue='baby')
    plt.show()

def main():
    df = get_data('data/merged_landmarks_dist.csv')

    histo_predictors(df)

    # matrix_one(df)
    # cor_mat(df)


if __name__ == "__main__":
    main()
