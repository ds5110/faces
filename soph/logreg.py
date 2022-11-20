#

from read_data import get_data
from helpers import get_Xy, tt_split, plot_cm, class_report

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression



def logreg(X,y):
    y_flat = np.ravel(y)
    Xtrain, Xtest, ytrain, ytest = tt_split(X,y_flat)
    logreg = LogisticRegression()
    fitted = logreg.fit(Xtrain, ytrain)
    y_pred = fitted.predict(Xtest)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(Xtest, ytest)))

    return y_pred, ytest, fitted

def main():
    df = get_data()
    #selecting certain columns as predictors:
    df_predictors = df.loc[:, ['image_name','boxratio', 'interoc','interoc','interoc_norm','boxsize','boxsize/interoc','baby']]
    
    X,y = get_Xy(df_predictors)

    y_pred, ytest, fitted = logreg(X,y)

    plot_cm(ytest,y_pred,'logreg')

    class_report(ytest,y_pred,'logreg')

if __name__ == "__main__":
    main()