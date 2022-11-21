#Sophia Cofone 11/19 
#File is intended for implementing logistic regression

from helpers import tt_split

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold


def logreg(X,y):
    y_flat = np.ravel(y)
    Xtrain, Xtest, ytrain, ytest = tt_split(X,y_flat)
    logreg = LogisticRegression(max_iter=1000)
    fitted = logreg.fit(Xtrain, ytrain)
    y_pred = fitted.predict(Xtest)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(Xtest, ytest)))

    return Xtest, ytest, fitted, y_pred

def rec_feature_selection(X,y):
    y_flat = np.ravel(y)
    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=1000),
        step=1,
        cv=StratifiedKFold(2),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X, y_flat)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print(f"features selected: {rfecv.ranking_}")
    print(rfecv.support_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()

    return rfecv.support_

def fwd_feature_selection(X,y,cols,n_features):
    y = np.ravel(y)
    estimator = LogisticRegression(max_iter=1000)
    fwdfs = SequentialFeatureSelector(estimator=estimator, 
    n_features_to_select=n_features, 
    n_jobs = -1)

    fwdfs.fit(X, y)

    selected_features = [i for (i, v) in zip(cols,list(fwdfs.get_support())) if v]
    return selected_features
