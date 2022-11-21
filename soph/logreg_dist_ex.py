#Sophia Cofone 11/20 
#File is intended for testing logistic regression with distance data (dif feature selection method)
#testing

from read_data import get_data, get_categories
from helpers import get_Xy, plot_cm, class_report
from imbalanced_resampling import upsample, downsample
from logreg import logreg, fwd_feature_selection

import numpy as np

def test_logreg_dist(df,predictors_list,desired_feaatures):
    print('With feature selection')
    sel_feat_lists = []
    accuracy = []
    desired_feaatures = desired_feaatures+1
    feature_nums = np.arange(1,desired_feaatures)

    for n in feature_nums:
        X,y = get_Xy(df,predictors_list)
        y = np.ravel(y)
        selected = fwd_feature_selection(X,y,predictors_list,n_features = n)
        sel_feat_lists.append(selected)
        
        X,y = get_Xy(df,selected)
        y = np.ravel(y)
        Xtest, ytest, fitted, y_pred = logreg(X,y)
        acc = fitted.score(Xtest,ytest)
        accuracy.append(acc)
        print("Features: {} , Accuracy: {}".format(n,acc))
        print(selected)
        print('')

        print('With feature selection and downsampled')
        downsample_df = downsample(df)
        X,y = get_Xy(downsample_df,selected)
        y = np.ravel(y)
        Xtest, ytest, fitted, y_pred = logreg(X,y)
        acc = fitted.score(Xtest,ytest)
        accuracy.append(acc)
        print("Features: {} , Accuracy: {}".format(n,acc))
        print(selected)
        print('')
        class_report(ytest,y_pred,'logreg')

    # def test_distbox():



def main():
    df = get_data('soph/merged_landmarks_dist.csv')
    _,_,_,_,_,all_d_coords = get_categories(df)

    print('All coordinates')
    test_logreg_dist(df,all_d_coords,4)

    print('Heuristically chosen coordinates')
    special_dist = {'LBW':'17_21','RBW':'22_26','innerB_dist':'21_22','outerB_dist':'17_26','LEW':'36_39','REW':'42_45',
     'innerE_dist':'39_42','outerE_dist':'36_45','LE_to_LB':'19_37','RE_to_RB':'24_44','Ear_to_Ear':'0_16',
     'ch_to_ch':'3_13','ch_to_nose':'8_30','MW':'48_54','MH':'57_62','Mcent_to_Ecent':'27_62',
     'Ncent_to_LE':'30_41','Ncent_to_RE':'30_46','Lip_to_LB':'19_62','Lip_to_RB':'24_62','Lchin_to_REar':'5_16',
     'Lchin_to_Mcent':'5_62','Lchin_to_N':'5_30','Lchin_to_LE':'5_37','Lchin_to_RE':'5_24','ch_to_M':'8_62'}
    special_dist_ = {key:'dist_'+value for (key,value) in special_dist.items()}
    predictors_list = list(special_dist_.values()) 
    test_logreg_dist(df,predictors_list,4)



if __name__ == "__main__":
    main()
