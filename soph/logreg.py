#Sophia Cofone 11/19 
#File is intended for trying out logistic regression

from read_data import get_data, get_categories
from helpers import get_Xy, tt_split, plot_cm, class_report
from imbalanced_resampling import upsample, downsample

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold




def logreg(X,y):
    y_flat = np.ravel(y)
    Xtrain, Xtest, ytrain, ytest = tt_split(X,y_flat)
    logreg = LogisticRegression()
    fitted = logreg.fit(Xtrain, ytrain)
    y_pred = fitted.predict(Xtest)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(Xtest, ytest)))

    return y_pred, ytest, fitted

def feature_selection(X,y):
    y_flat = np.ravel(y)
    min_features_to_select = 1  # Minimum number of features to consider
    rfecv = RFECV(
        estimator=LogisticRegression(),
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

def main():
    df = get_data()
    main_predictors, norm_cenrot_sym_diff = get_categories(df)
    print(norm_cenrot_sym_diff)
    #selecting certain columns as predictors:
    predictors_list = norm_cenrot_sym_diff
    # predictors_list = ['norm_cenrot_sym_diff-x36', 'norm_cenrot_sym_diff-y36', 'norm_cenrot_sym_diff-x39', 'norm_cenrot_sym_diff-y39', 'norm_cenrot_sym_diff-x37', 'norm_cenrot_sym_diff-y37', 'norm_cenrot_sym_diff-x38', 'norm_cenrot_sym_diff-y38', 'norm_cenrot_sym_diff-x40', 'norm_cenrot_sym_diff-y40', 'norm_cenrot_sym_diff-x41', 'norm_cenrot_sym_diff-y41', 'norm_cenrot_sym_diff-x31', 'norm_cenrot_sym_diff-y31', 'norm_cenrot_sym_diff-x32', 'norm_cenrot_sym_diff-y32', 'norm_cenrot_sym_diff-x0', 'norm_cenrot_sym_diff-y0', 'norm_cenrot_sym_diff-x1', 'norm_cenrot_sym_diff-y1', 'norm_cenrot_sym_diff-x2', 'norm_cenrot_sym_diff-y2', 'norm_cenrot_sym_diff-x3', 'norm_cenrot_sym_diff-y3', 'norm_cenrot_sym_diff-x4', 'norm_cenrot_sym_diff-y4', 'norm_cenrot_sym_diff-x5', 'norm_cenrot_sym_diff-y5', 'norm_cenrot_sym_diff-x6', 'norm_cenrot_sym_diff-y6', 'norm_cenrot_sym_diff-x7', 'norm_cenrot_sym_diff-y7', 'norm_cenrot_sym_diff-x17', 'norm_cenrot_sym_diff-y17', 'norm_cenrot_sym_diff-x18', 'norm_cenrot_sym_diff-y18', 'norm_cenrot_sym_diff-x19', 'norm_cenrot_sym_diff-y19', 'norm_cenrot_sym_diff-x20', 'norm_cenrot_sym_diff-y20', 'norm_cenrot_sym_diff-x21', 'norm_cenrot_sym_diff-y21', 'norm_cenrot_sym_diff-x48', 'norm_cenrot_sym_diff-y48', 'norm_cenrot_sym_diff-x49', 'norm_cenrot_sym_diff-y49', 'norm_cenrot_sym_diff-x50', 'norm_cenrot_sym_diff-y50', 'norm_cenrot_sym_diff-x60', 'norm_cenrot_sym_diff-y60', 'norm_cenrot_sym_diff-x61', 'norm_cenrot_sym_diff-y61', 'norm_cenrot_sym_diff-x67', 'norm_cenrot_sym_diff-y67', 'norm_cenrot_sym_diff-x59', 'norm_cenrot_sym_diff-y59', 'norm_cenrot_sym_diff-x58', 'norm_cenrot_sym_diff-y58']
    
    # print('with without feature selection')
    # X,y = get_Xy(df,predictors_list)
    # y_pred, ytest, fitted = logreg(X,y)
    # plot_cm(ytest,y_pred,'logreg')
    # class_report(ytest,y_pred,'logreg')

    #trying logreg with feature selection
    #feature selection
    X,y = get_Xy(df,predictors_list)
    bool_features = feature_selection(X,y)
    selected_features = [i for (i, v) in zip(predictors_list,list(bool_features)) if v]

    # print('with feature selection (2)')
    # #logreg
    # predictors_list = selected_features
    # X,y = get_Xy(df,predictors_list)
    # y_pred, ytest, fitted = logreg(X,y)
    # plot_cm(ytest,y_pred,'logreg')
    # class_report(ytest,y_pred,'logreg')

    # #logreg
    # print('with feature selection (1)')
    # predictors_list = [selected_features[0]]
    # X,y = get_Xy(df,predictors_list)
    # y_pred, ytest, fitted = logreg(X,y)
    # plot_cm(ytest,y_pred,'logreg')
    # class_report(ytest,y_pred,'logreg')



    #logreg with feature selection and downsampled
    predictors_list = selected_features
    upsample_df = downsample(df)
    X,y = get_Xy(upsample_df,predictors_list)
    y_pred, ytest, fitted = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')



if __name__ == "__main__":
    main()