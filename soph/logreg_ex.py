#Sophia Cofone 11/20 
#File is intended for testing logistic regression

from read_data import get_data, get_categories
from helpers import get_Xy, plot_cm, class_report
from imbalanced_resampling import upsample, downsample
from logreg import logreg, rec_feature_selection

def test_partition_a(df,predictors_list):
    print('Without feature selection')
    X,y = get_Xy(df,predictors_list)
    _, ytest, _, y_pred = logreg(X,y)
    plot_cm(ytest,y_pred,'logreg')
    class_report(ytest,y_pred,'logreg')
    
    print('Feature selection')
    X,y = get_Xy(df,predictors_list)
    selected_features = rec_feature_selection(X,y,predictors_list)

    return selected_features
    
def test_partition_b(df,selected_features):
    print('With feature selection')
    predictors_list = selected_features
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

def main():
    df = get_data()
    _, angle_off, main_predictors,norm_cenrot_sym_diff, norm_cenrot,_,_ = get_categories(df)

    print('partition 1')
    selected_features = test_partition_a(df,main_predictors) 
    test_partition_b(df,selected_features)

    print('partition 2')
    selected_features = test_partition_a(df,main_predictors) 
    test_partition_b(df,selected_features)
    #choosing 1
    selected_features = selected_features[0:1]
    test_partition_b(df,selected_features)

    print('partition 3')
    selected_features = test_partition_a(df,main_predictors+angle_off) 
    #choosing 2
    selected_features = selected_features[0:2]
    test_partition_b(df,selected_features)

    print('partition 4')
    selected_features = test_partition_a(df,norm_cenrot) 
    #choosing 66
    selected_features = selected_features[0:66]
    test_partition_b(df,selected_features)

    print('partition 5')
    selected_features = test_partition_a(df,norm_cenrot_sym_diff)
    test_partition_b(df,selected_features)

if __name__ == "__main__":
    main()
