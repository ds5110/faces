#Sophia Cofone 11/21/22
'''
File is intended for implementing forward and backward (recursive) feature selection
'''
#project
from util.sc_helpers import tt_split
#basic
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
#sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import make_pipeline

def tune_rfe(estimator,num_features,n_folds,X,y):
    '''
    Tuning the hyperparameter number of features for RFE.
    Returns scores and the index referring to the number of selected features.
    '''
    y_flat = np.ravel(y)
    rfe = make_pipeline(RFE(estimator=estimator))
    tuned_parameters = [{"rfe__n_features_to_select": num_features}]
    clf = GridSearchCV(rfe, tuned_parameters, cv=n_folds, refit=False, return_train_score=True)
    clf.fit(X, y_flat)
    scores = clf.cv_results_["mean_test_score"]
    scores_std = clf.cv_results_["std_test_score"]
    scores_train = clf.cv_results_["mean_train_score"]
    scores_std_train = clf.cv_results_["std_train_score"]

    num_features_selected = clf.best_index_+1
    print('Number of features selected by RFE: {:.2f}'.format(num_features_selected))
   
    return scores,scores_std,scores_train,scores_std_train,num_features_selected

def plot_tune(scores,scores_std,scores_train,scores_std_train,num_features,n_folds):
    '''
    Plotting function for the gridsearch.
    '''
    #plotting
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    
    ax.plot(num_features, scores)
    ax.plot(num_features, scores_train)

    std_error = scores_std / np.sqrt(n_folds)
    ax.plot(num_features, scores + std_error, "b--")
    ax.plot(num_features, scores - std_error, "b--")

    std_error_train = scores_std_train / np.sqrt(n_folds)
    ax.plot(num_features, scores_train + std_error_train, linestyle=":", color='orange')
    ax.plot(num_features, scores_train - std_error_train, linestyle=":",color='orange')

    # alpha=0.2 controls the translucency of the fill color
    ax.fill_between(num_features, scores + std_error, scores - std_error, alpha=0.2)
    ax.fill_between(num_features, scores_train + std_error_train, scores_train - std_error_train, alpha=0.2)
    plt.ylabel("CV score +/- std error")
    plt.xlabel("Number of Features")
    plt.axhline(np.max(scores), linestyle="--", color=".5")
    plt.axhline(np.max(scores_train), linestyle=":", color=".5")
    plt.xlim([num_features[0], num_features[-1]])

    plt.title("CV scores vs number of features used")
    blue_patch = mpatches.Patch(color='blue', label='Training')
    orange_patch = mpatches.Patch(color='orange', label='Testing')

    ax.legend(handles=[orange_patch,blue_patch])

    plt.show()

def rec_feature_selection(estimator,num_features,X,y,predictors_list):
    '''
    Does RFE.
    Returns the train test sets, and the selected features. 
    '''
    y_flat = np.ravel(y)
    selector = RFE(estimator, n_features_to_select=num_features)
    Xtrain, Xtest, ytrain, ytest = tt_split(X,y_flat)
    fitted = selector.fit(Xtrain, ytrain)
    y_pred = fitted.predict(Xtest)

    selected_features = [i for (i, v) in zip(predictors_list,list(fitted.get_support())) if v]

    return Xtrain, Xtest, ytrain, ytest, y_pred, selected_features

def fwd_feature_selection(X,y,predictors_list,n_features, estimator=LogisticRegression()):
    '''
    Forward feature selection.
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
