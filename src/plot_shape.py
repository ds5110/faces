#Sophia Cofone, 10/24/22, file is intended to plot shape of images for EDA

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

'''Thought process behind this file:
Goal: I wanted to see if the images were all the same size (same number of rows and columns) 
so I could find out if I needed to scale the values for further EDA
Result: They are the same size, see the histogram and scatterplot
'''

path = './data/'
file_name = path + 'labels.csv'

#Creating dataframe
df = pd.DataFrame(pd.read_csv(file_name))

#Creating path to the specific image
def get_im_path(im_num,df):
    return os.path.join(path+'images',
                           df['image-set'].iloc[im_num],df['filename'].iloc[im_num])

#Going through all the images and finding their shape
def get_img_shapes():
    dim_x = []
    dim_y = []
    for i in range(df.shape[0]):
        im_path = get_im_path(i,df)
        image = mpimg.imread(im_path)
        dim_x,dim_y = (image.shape[0],image.shape[1])
    df['dim_x'] = dim_x
    df['dim_y'] = dim_y

get_img_shapes()

#Plotting result
plt.hist(df['dim_x'])
plt.show()
plt.scatter(df['dim_x'],df['dim_y'])
plt.show()


