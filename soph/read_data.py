#Sophia Cofone 11/19 
#File is intended to read df

import pandas as pd

def get_data():
    '''Gets the whole data set, seperates into adult and baby df, adds the class column, and outputs concatenated df'''
    
    df = pd.read_csv('data/merged_landmarks.csv',dtype={
        'image-set': str,
        'filename': str,
        'partition': str,
        'subpartition': str,})

    return df

def get_categories(df):
    main_predictors = ['boxratio', 'interoc','interoc_norm','boxsize','boxsize/interoc']
    df1 = df.loc[:, df.columns.str.startswith('norm_cenrot_sym_diff')]
    norm_cenrot_sym_diff = list(df1.columns)
    # dim = 'x'
    # norm_cenrot_sym_diff=['norm_cenrot_sym_diff-{}{}'.format(dim,i) for i in range(67)]
    # dim = 'y'
    # norm_cenrot_sym_diff = norm_cenrot_sym_diff+['norm_cenrot_sym_diff-{}{}'.format(dim,i) for i in range(68)]
    
    return main_predictors, norm_cenrot_sym_diff

def main():
    df = get_data()
    main_predictors, norm_cenrot_sym_diff = get_categories(df)

if __name__ == "__main__":
    main()