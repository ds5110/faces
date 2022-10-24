#Sophia Cofone, 10/24/22, file is intended to plot annotations onto images for EDA

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

'''Thought process behind this file:
Goal: Visualize the annotations
Result: 3 plots. 1 plot of all the annotations, 1 plot of a single image's annotations, and 1 plot of a single image + its annotations
It seems like even though images are same shape, the babies are different sizes 
'''

path = './data/'
file_name = path + 'labels.csv'

def get_im_path(im_num,df):
    return os.path.join(path+'images',
                           df['image-set'].iloc[im_num],df['filename'].iloc[im_num])

#Creating dataframe
df = pd.DataFrame(pd.read_csv(file_name))

#plotting all the feature (f) annotations in the data set
def plot_all_annotations(df):
    for f in range(67):
        xcol = df['gt-x{}'.format(f)]
        ycol = df['gt-y{}'.format(f)]
        plt.scatter(xcol,ycol)
    plt.show()

#plotting all the feature (f) annotations in a single image (row) of data set
def plot_single_annotations(im_num,df):
    for f in range(67):
        x = df['gt-x{}'.format(f)][im_num]
        y = df['gt-y{}'.format(f)][im_num]
        plt.scatter(x,y)
    plt.show()

#plotting all the feature (f) annotations in a single image (row) of data set on top of the actual image
def plot_single_annotations_baby(im_num,df):
    image_fpath = get_im_path(im_num,df)
    image = mpimg.imread(image_fpath)
    plt.imshow(image)
    plot_single_annotations(im_num,df)
    plt.show()

plot_all_annotations(df)
plot_single_annotations(0,df)
plot_single_annotations_baby(0,df)
        