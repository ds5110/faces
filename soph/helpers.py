#

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#Splits the data into X(data) and y(target)
def get_Xy(df):
    X = df.iloc[:, 1:-1]
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