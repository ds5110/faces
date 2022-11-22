#Sophia Cofone 11/19 
#File is intended to house some helper funtions that multiple models will use

#basic
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
#sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def get_data(data='data/merged_landmarks.csv'):    
    df = pd.read_csv(data,dtype={
        'image-set': str,
        'filename': str,
        'partition': str,
        'subpartition': str,})
    return df

#Splits the data into X(data) and y(target)
def get_Xy(df,predictors=None):
    if predictors is not None:
        in_list = predictors+['baby']
        df = df.loc[:, in_list]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    X = X.to_numpy()
    y = y.to_numpy()
    return X,y

# Train/test split
def tt_split(X,y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)
    
    return Xtrain, Xtest, ytrain, ytest

# Plot confusion matrix
def plot_cm(ytest,y_model,title):
    mat = confusion_matrix(ytest, y_model)
    sns.heatmap(mat, square=True, annot=True, cbar=False, cmap='OrRd')
    plt.title(title)
    plt.xlabel('predicted value')
    plt.ylabel('true value')
    plt.show()

#create classification report
def class_report(y_test,y_model,title):
    print(
    f"Classification report for classifier {title}:\n"
    f"{classification_report(y_test, y_model)}\n"
)

#defining some additional categories
def get_categories(df):

    #Difference of expected symmetric landmarks, yaw-corrected coordinates, normalized per min box
    df1 = df.loc[:, df.columns.str.startswith('norm_cenrot_sym_diff')]
    norm_cenrot_sym_diff = list(df1.columns)

    #Euc distance of all coords (yaw and box corrected)
    #all of them
    df1 = df.loc[:, df.columns.str.startswith('dist_')]
    all_d_coords = list(df1.columns)

    return norm_cenrot_sym_diff,all_d_coords