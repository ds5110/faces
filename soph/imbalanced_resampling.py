#Sophia Cofone 11/18/22

'''
We need to address the fact that we have 3147 points in our adult class (0) and  209 points in our baby class (1).
There are several ways to "deal" with unbalanced data such as Up-sampling and undersampling.
reference https://elitedatascience.com/imbalanced-classes
'''

from read_data import get_data, get_Xy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics




def show_imbalance(data):
    sns.countplot(x='class', data=data)
    plt.show()

def upsample(df_adult,df_baby):
    '''Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.'''
    #Resample with replacement
    print(df_adult['class'].value_counts())
    #Separate majority and minority classes
    df_majority = df_adult
    df_minority = df_baby
    #Upsample minority class
    df_minority_upsampled = resample(df_minority, 
    replace=True,
    n_samples=3147,#to match majority class
    random_state=42)
    
    #Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    counts = df_upsampled['class'].value_counts()
    print(f'Resulting instances of class 0 and 1 after upsampling: {counts}')

    return df_upsampled

def downsample(df_adult,df_baby):
    #Separate majority and minority classes
    df_majority = df_adult
    df_minority = df_baby
    #Downsample majority class
    df_majority_downsampled = resample(df_majority, 
    replace=False,
    n_samples=209,# to match minority class
    random_state=42) 
    
    #Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    counts = df_downsampled['class'].value_counts()
    print(f'Resulting instances of class 0 and 1 after upsampling: {counts}')

    return df_downsampled

# Train/test split
def tt_split(X,y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
    
    return Xtrain, Xtest, ytrain, ytest

# Train and test the model
def gnb_model(Xtrain,ytrain,Xtest,ytest):
    model = GaussianNB()
    model.fit(Xtrain, ytrain)
    y_model = model.predict(Xtest)
    print('\nTrain accuracy: {:.2f}'.format(accuracy_score(ytrain, model.predict(Xtrain))))
    print('Test accuracy: {:.2f}\n'.format(accuracy_score(ytest, y_model)))
    return y_model


def logisticr(Xtrain,ytrain,Xtest,ytest):
    logreg = LogisticRegression()
    logreg.fit(Xtrain, ytrain)
    y_pred = logreg.predict(Xtest)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(Xtest, ytest)))




'''
We can also look at specific preformance metrics like AROC
'''



# def AUROC():
# prob_y_2 = clf_2.predict_proba(X)
 
# # Keep only the positive class
# prob_y_2 = [p[1] for p in prob_y_2]
 
# prob_y_2[:5] # Example
# # [0.4515319725758555,
# #  0.48726124480997834,
# #  0.47238960854127,
# #  0.4701461062264753,
# #  0.5876602955884178]

def main():
    df_adult, df_baby, combine_df = get_data()
    show_imbalance(combine_df)
    upsample_df = upsample(df_adult,df_baby)
    downsample_df = downsample(df_adult,df_baby)

    X,y = get_Xy(upsample_df)
    Xtrain, Xtest, ytrain, ytest = tt_split(X,y)
    y_model = logisticr(Xtrain,ytrain,Xtest,ytest)
    # y_model = gnb_model(Xtrain,ytrain,Xtest,ytest)

if __name__ == "__main__":
    main()