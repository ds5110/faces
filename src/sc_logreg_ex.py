#Sophia Cofone 11/20 
'''
This file is intended for testing logistic regression model.
'''
#project
from util.sc_helpers import get_Xy, plot_cm, class_report, tt_split, get_data, get_categories
from util.sc_resample import upsample, downsample
from util.sc_feature_selection import tune_rfe, plot_tune, rec_feature_selection, fwd_feature_selection
from util.model import main_predictors
from util.column_names import cenrot_cols
#basic
import numpy as np
#sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def logreg(X,y):
    '''
    Simple logreg function 
    '''
    y_flat = np.ravel(y)
    Xtrain, Xtest, ytrain, ytest = tt_split(X,y_flat)
    logreg = LogisticRegression(max_iter=1000)
    fitted = logreg.fit(Xtrain, ytrain)
    y_pred = fitted.predict(Xtest)
    print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(Xtest, ytest)))

    return Xtrain, Xtest, ytrain, ytest, fitted, y_pred

def without_f(df,predictors_list):
    X,y = get_Xy(df,predictors_list)
    Xtrain, Xtest, ytrain, ytest, fitted, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')

def rec_feature_tune(df,predictors_list):
    X,y = get_Xy(df,predictors_list)
    num_features = np.arange(1,len(predictors_list)+1)
    num_folds = 5   
    estimator = LogisticRegression(max_iter=1000)
    scores,scores_std,scores_train,scores_std_train,num_features_selected = tune_rfe(estimator,num_features,num_folds,X,y,predictors_list)
    plot_tune(scores,scores_std,scores_train,scores_std_train,num_features,num_folds)

    return num_features_selected

def rec_feature_select(df,predictors_list,num_features):
    X,y = get_Xy(df,predictors_list)
    Xtrain, Xtest, ytrain, ytest, y_pred, selected_features = rec_feature_selection(LogisticRegression(max_iter=1000),num_features,X,y,predictors_list)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')
    print(selected_features)

def fwd_feature_select(df,n_features,predictors_list):
    X,y = get_Xy(df,predictors_list)
    estimator = LogisticRegression(max_iter=1000)
    Xtrain, Xtest, ytrain, ytest, y_pred, selected_features = fwd_feature_selection(X,y,predictors_list,n_features, estimator)    
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')
    print(selected_features)





def main():
    df = get_data()
    norm_cenrot_sym_diff,_ = get_categories(df)

    # print('Partition 1 - Without feature selection')
    # without_f(df,main_predictors) 
    # print('Partition 1 - Feature selection')
    # num_features_selected = rec_feature_tune(df,main_predictors)
    # print('Partition 1 - With Feature selection')
    # rec_feature_select(df,main_predictors,num_features_selected)

    print('Partition 1 - Without feature selection')
    without_f(df,main_predictors) 
    print('Partition 1 - Feature selection')
    num_features_selected = fwd_feature_select(df,3,main_predictors)
    print('Partition 1 - With Feature selection')
    # rec_feature_select(df,main_predictors,num_features_selected)
    
    # print('Partition 2 - Without feature selection')
    # without_f(df,cenrot_cols) 
    # print('Partition 2 - Feature selection')
    # num_features_selected = rec_feature_tune(df,cenrot_cols)
    # print('Partition 2 - With Feature selection')
    # rec_feature_select(df,main_predictors,num_features_selected)

    # print('Testing out box + eucdistances')
    # df = get_data('soph/merged_landmarks_dist.csv')
    # predictors_list = ['boxratio','dist_7_41','dist_21_22', 'dist_22_25', 'dist_33_65']
    # with_f(df,predictors_list)

if __name__ == "__main__":
    main()
