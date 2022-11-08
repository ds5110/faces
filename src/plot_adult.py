#Sophia Cofone, 11/7/22, file is intended to plot annotations onto images for EDA

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

'''Thought process behind this file:
Goal: Visualize the annotations
Result: 2 plots. 1 plot of all the annotations, 1 plot of a single image's annotations.
'''

path = './data/'
file_name = path + '300w_valid.csv'

# def get_im_path(im_num,df):
#     return os.path.join(path+'images',
#                            df['image-set'].iloc[im_num],df['filename'].iloc[im_num])

#Creating dataframe
df = pd.DataFrame(pd.read_csv(file_name))

#plotting all the feature (f) annotations in the data set
def plot_all_annotations_adult(df):
    for f in range(67):
        xcol = df['original_{}_x'.format(f)]
        ycol = df['original_{}_y'.format(f)]
        plt.scatter(xcol,ycol)
    plt.show()

#plotting all the feature (f) annotations in a single image (row) of data set
def plot_single_annotations_adult(im_num,df):
    for f in range(67):
        x = df['original_{}_x'.format(f)][im_num]
        y = df['original_{}_y'.format(f)][im_num]
        plt.scatter(x,y)
    plt.show()

# #plotting all the feature (f) annotations in a single image (row) of data set on top of the actual image
# def plot_single_annotations_baby(im_num,df):
#     image_fpath = get_im_path(im_num,df)
#     image = mpimg.imread(image_fpath)
#     plt.imshow(image)
#     plot_single_annotations(im_num,df)
#     plt.show()

plot_all_annotations_adult(df)
plot_single_annotations_adult(0,df)
# plot_single_annotations_baby(0,df)
        