#Sophia Cofone 11/20 
'''
This file is intended for testing logistic regression model.
'''
#project
from read_data import get_data, get_categories
from helpers import get_Xy, plot_cm, class_report
from imbalanced_resampling import upsample, downsample
from logreg import logreg, rec_feature_selection

def without_f(df,predictors_list):
    print('Without feature selection')
    X,y = get_Xy(df,predictors_list)
    _, ytest, _, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')

def with_f(df,predictors_list):
    print('Feature selection')
    X,y = get_Xy(df,predictors_list)
    selected_features = rec_feature_selection(X,y,predictors_list)

    print('With feature selection')
    predictors_list = selected_features[0:None]
    X,y = get_Xy(df,predictors_list)
    _, ytest, _, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')

    print('With feature selection and downsampled')
    upsample_df = downsample(df)
    X,y = get_Xy(upsample_df,predictors_list)
    _, ytest, _, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')

    return selected_features

def with_f_choice(df,selected_features,choice):
    print('With feature selection (choice)')
    predictors_list = selected_features[0:choice]
    X,y = get_Xy(df,predictors_list)
    _, ytest, _, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')

    print('With feature selection (choice) and downsampled')
    upsample_df = downsample(df)
    X,y = get_Xy(upsample_df,predictors_list)
    _, ytest, _, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')

def main():
    df = get_data()
    _, angle_off, main_predictors,norm_cenrot_sym_diff, norm_cenrot,_ = get_categories(df)

    print('Partition 1 - Without feature selection')
    without_f(df,main_predictors) 
    print('Partition 1 - With feature selection')
    selected_features = with_f(df,main_predictors)

    print('Partition 2 - With feature selection (choice)')
    #choosing 1
    with_f_choice(df,selected_features,1)
    
    print('Partition 3 - Without feature selection')
    without_f(df,main_predictors+angle_off) 
    print('Partition 3 - With feature selection')
    selected_features = with_f(df,main_predictors+angle_off)
    #choosing 2
    print('Partition 3 - With feature selection (choice)')
    with_f_choice(df,selected_features,2)

    print('Partition 4 - Without feature selection')
    without_f(df,norm_cenrot) 
    print('Partition 4 - With feature selection')
    selected_features = with_f(df,norm_cenrot)
    #choosing 66
    print('Partition 4 - With feature selection (choice)')
    with_f_choice(df,selected_features,66)

    print('Partition 5 - Without feature selection')
    without_f(df,norm_cenrot_sym_diff) 
    print('Partition 5 - With feature selection')
    selected_features = with_f(df,norm_cenrot_sym_diff)

if __name__ == "__main__":
    main()
