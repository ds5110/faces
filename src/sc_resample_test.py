#Sophia Cofone 11/18/22
'''
File is intended to test resampling functions.
'''
#project
from util.sc_helpers import get_data
from util.sc_helpers  import get_Xy, class_report
from util.sc_resample import upsample, downsample

def main():
    from sc_logreg_ex import logreg
    #testing out the different sampling options with logreg
    #get data
    df = get_data()
    predictors_list = ['boxratio']

    #testing
    print('Trying with no sampling changes (unbalanced)')
    X,y = get_Xy(df,predictors_list)
    Xtest, ytest, fitted, y_pred = logreg(X,y)
    class_report(ytest,y_pred,'logreg')

    print('Trying with upsampling')
    upsample_df = upsample(df)
    X,y = get_Xy(upsample_df,predictors_list)
    Xtest, ytest, fitted, y_pred = logreg(X,y)
    class_report(ytest,y_pred,'logreg')

    print('Trying with downsampling')
    downsample_df = downsample(df)
    X,y = get_Xy(downsample_df,predictors_list)
    Xtest, ytest, fitted, y_pred = logreg(X,y)
    class_report(ytest,y_pred,'logreg')

if __name__ == "__main__":
    main()