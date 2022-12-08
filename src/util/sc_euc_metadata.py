#Sophia Cofone 11/20/22
'''
This file is intended to calculate additional meta-data for computing the euclidean distances between the coordinate points.
We use the normalized to the bounding box data as the coord input, then compute the distance between each point and every other point.
The output is an updated df, but the  "main" function below puts it into a CSV.
'''

#project
from sc_helpers import get_data
#basics
import pandas as pd
import numpy as np
#sklearn
from sklearn.metrics.pairwise import euclidean_distances as edist

def get_dist_df(df_cr,f_num_list = range(68)):
    """ 
    df input should be a subet of the main one. Below I used the "norm_cenrot-" derived columns.
    f_num_list = the list of feature (coordinates) numbers to compute the distances betweeen, 
    ex. range(68) for all of them, or [1,5,6,20,30,55]
    """

    #Create the headers for the distances. Need to make sure not to include distance between f1-f2 and f2-f1 (duplicates).
    dist_labels = []
    for i in f_num_list:
        for j in f_num_list:
            if ('dist_{}_{}'.format(i,j) not in dist_labels) and (('dist_{}_{}'.format(j,i) not in dist_labels)) and i!=j:
                dist_labels.append('dist_{}_{}'.format(i,j))

    #Create matrix with row=images,cols=distances
    num_feat = len(f_num_list)
    im_dist_mat = np.zeros((df_cr.shape[0],int(num_feat*(num_feat-1)/2)))

    for image_index in range(df_cr.shape[0]):
        anot_data = df_cr.iloc[image_index]
        anot_coords = []
        for coord in f_num_list:
            x = anot_data['norm_cenrot-x{}'.format(coord)]
            y = anot_data['norm_cenrot-y{}'.format(coord)]
            anot_coords.append(np.array([x,y]))

        #Get the pairwise euclidian distance between all points
        dist_mat = edist(anot_coords)
        #Get a flattened array of the pairwise distances, remove the diagonal since self distances are 0
        dist_list = dist_mat[np.triu_indices(dist_mat.shape[0],k = 1)]
        #Insert the distances for this image into the master matrix
        im_dist_mat[image_index,:] = dist_list
        
    # create and return the data_frame of the distances
    return pd.DataFrame(im_dist_mat,columns = dist_labels)

def main():
    #grabbing main DF
    df = get_data()
    #getting subset of DF
    df_cr = df.loc[:,df.columns.str.startswith('norm_cenrot-')]
    #calculating new df
    df_dist = get_dist_df(df_cr,f_num_list = range(68))
    #adding it to the main df and outputting to csv
    df = pd.concat([df.copy(),df_dist],axis=1)
    df.to_csv('data/merged_landmarks_dist.csv')

if __name__ == "__main__":
    main()