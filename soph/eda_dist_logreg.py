
from read_data import get_data

import numpy as np
import matplotlib.pyplot as plt

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


def main():
    df = get_data('data/merged_landmarks_dist.csv')
    matrix_one(df)
    cor_mat(df)

if __name__ == "__main__":
    main()
