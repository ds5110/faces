#Sophia Cofone 11/18/22
'''
File is intended to house functions used for up and downsampling.

We need to address the fact that we have 689 points in our adult class (0) and  410 points in our baby class (1).
This is not a wild imbalance, but ideally for classificiation tasks we would have balanced dataset.
There are several ways to "deal" with unbalanced data such as up-sampling and undersampling.
We will use logistic regression to see the differences in the sampling methods (for example).

reference https://elitedatascience.com/imbalanced-classes
'''
#project
from read_data import get_data
from helpers import get_Xy, class_report
#basic
import pandas as pd
#sklearn
from sklearn.utils import resample

def split_df(df):
    df_baby = df[df['baby']==1]
    df_adult = df[df['baby']==0]

    return df_baby, df_adult

def upsample(df):
    '''Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.'''
    df_baby, df_adult = split_df(df)
    
    #Resample with replacement
    print(df_adult['baby'].value_counts())
    #Separate majority and minority classes
    df_majority = df_adult
    df_minority = df_baby
    #Upsample minority class
    df_minority_upsampled = resample(df_minority, 
    replace=True,
    n_samples=689,#to match majority class
    random_state=42)
    
    #Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    counts = df_upsampled['baby'].value_counts()
    print(f'Resulting instances of class 0 and 1 after upsampling: {counts}')

    return df_upsampled

def downsample(df):
    '''randomly removes elements of the majority class'''
    df_baby, df_adult = split_df(df)

    #Separate majority and minority classes
    df_majority = df_adult
    df_minority = df_baby
    #Downsample majority class
    df_majority_downsampled = resample(df_majority, 
    replace=False,
    n_samples=410,# to match minority class
    random_state=42) 
    
    #Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    counts = df_downsampled['baby'].value_counts()
    print(f'Resulting instances of class 0 and 1 after upsampling: {counts}')

    return df_downsampled
    
def main():
    from logreg import logreg
    #testing out the different sampling options with logreg
    #get data
    df = get_data()
    predictors_list = ['boxratio']

    #testing
    print('Trying with no sampling changes (unbalanced)')
    X,y = get_Xy(df,predictors_list)
    y_pred, ytest, fitted = logreg(X,y)
    class_report(ytest,y_pred,'logreg')

    print('Trying with upsampling')
    upsample_df = upsample(df)
    X,y = get_Xy(upsample_df,predictors_list)
    y_pred, ytest, fitted = logreg(X,y)
    class_report(ytest,y_pred,'logreg')

    print('Trying with downsampling')
    downsample_df = downsample(df)
    X,y = get_Xy(downsample_df,predictors_list)
    y_pred, ytest, fitted = logreg(X,y)
    class_report(ytest,y_pred,'logreg')

if __name__ == "__main__":
    main()