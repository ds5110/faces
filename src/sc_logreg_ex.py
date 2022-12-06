#Sophia Cofone 11/20 
'''
This file is intended for testing logistic regression model.
'''
#project
from util.sc_helpers import get_Xy, plot_cm, class_report, tt_split, get_data, get_categories
from util.sc_resample import upsample, downsample
from util.sc_feature_selection import tune_rfe, plot_tune, rec_feature_selection, fwd_feature_selection
from util.model import main_predictors, norm_cols

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
    scores,scores_std,scores_train,scores_std_train,num_features_selected = tune_rfe(estimator,num_features,num_folds,X,y)
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

    # print('Testing out [boxratio, interoc,interoc_norm,boxsize,boxsize/interoc]')
    # print('Partition 1 - Without feature selection')
    # without_f(df,main_predictors) 
    # print('Partition 1 - Feature selection')
    # num_features_selected = rec_feature_tune(df,main_predictors)
    # print('Partition 1 - With Feature selection')
    # rec_feature_select(df,main_predictors,num_features_selected)
    # print('Partition 1 - With Feature selection & downsampled')
    # df_downsampled = downsample(df)
    # rec_feature_select(df_downsampled,main_predictors,num_features_selected)

    # print('Testing out [norm_cenrot-]')
    # print('Partition 2 - Without feature selection')
    # without_f(df,norm_cols[0]) 
    # print('Partition 2 - Feature selection')
    # num_features_selected = rec_feature_tune(df,norm_cols[0])
    # print('Partition 2 - With Feature selection')
    rec_feature_select(df,norm_cols[0],29)
    print('Partition 2 - With Feature selection & downsampled')
    df_downsampled = downsample(df)
    rec_feature_select(df_downsampled,norm_cols[0],29)


    # print('Testing out boxratio + specific eucdistances')
    # df = get_data('data/merged_landmarks_dist.csv')
    # predictors_list = ['boxratio','dist_5_7', 'dist_7_9', 'dist_7_48', 'dist_18_25', 'dist_21_22', 'dist_33_42']
    # print('Partition 3 - Without feature selection')
    # without_f(df,predictors_list) 
    # print('Partition 3 - Feature selection')
    # num_features_selected = rec_feature_tune(df,predictors_list)
    # print('Partition 3 - With Feature selection')
    # rec_feature_select(df,predictors_list,num_features_selected)
    # df_downsampled = downsample(df)
    # rec_feature_select(df_downsampled,predictors_list,num_features_selected)


if __name__ == "__main__":
    main()
