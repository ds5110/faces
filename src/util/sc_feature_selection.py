#Sophia Cofone 11/21/22
'''
File is intended for implementing forward and backward (recursive) feature selection
'''
#basic
import matplotlib.pyplot as plt
import numpy as np
#sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold

def rec_feature_selection(X,y,predictors_list,estimator=LogisticRegression()):
    '''
    Recursive/backward feature selection with CV to see what the optimal number is.
    Produces what it thinks optimal is (max score) and plot so we can go in and choose a different number of features if needed.
    Returns a list of selected features.
    '''
    y_flat = np.ravel(y)
    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(2),
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
    )
    rfecv.fit(X, y_flat)

    print("Optimal number of features : %d" % rfecv.n_features_)
    print(f"features selected: {rfecv.ranking_}")

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (accuracy)")
    plt.plot(
        range(min_features_to_select, len(rfecv.grid_scores_) + min_features_to_select),
        rfecv.grid_scores_,
    )
    plt.show()
    
    selected_features = [i for (i, v) in zip(predictors_list,list(rfecv.support_)) if v]

    return selected_features

def fwd_feature_selection(X,y,predictors_list,n_features, estimator=LogisticRegression()):
    '''
    Forward feature selection.
    This one doesnt have CV implemented beucase it would take a really long time for it to run (for dimentional reduction of euclidian distances).
    Returns a list of selected features.
    '''
    y = np.ravel(y)
    estimator = estimator
    fwdfs = SequentialFeatureSelector(estimator=estimator, 
    n_features_to_select=n_features, 
    n_jobs = -1)

    fwdfs.fit(X, y)
    
    selected_features = [i for (i, v) in zip(predictors_list,list(fwdfs.get_support())) if v]

    return selected_features
