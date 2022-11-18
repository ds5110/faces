import pandas as pd


def get_data():
    infant = pd.read_csv('https://coe.northeastern.edu/Research/AClab/InfAnFace/labels.csv')
    joint = pd.read_csv('https://raw.githubusercontent.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/master/data/joint/300w_infanface_train.csv')
    adult = pd.read_csv('https://raw.githubusercontent.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/master/data/300w/300w_train.csv')

    infant.to_csv('infant')
    joint.to_csv('joint')
    adult.to_csv('adult')


get_data()