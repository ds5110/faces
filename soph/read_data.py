#

import pandas as pd

def get_data():
    '''Gets the whole data set, seperates into adult and baby df, adds the class column, and outputs concatenated df'''
    
    df = pd.read_csv('data/merged_landmarks.csv',dtype={
        'image-set': str,
        'filename': str,
        'partition': str,
        'subpartition': str,})

    return df

def main():
    df = get_data()

if __name__ == "__main__":
    main()