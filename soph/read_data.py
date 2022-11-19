import pandas as pd


def get_data():
    '''Gets the joint data set, seperates into adult and baby df, adds the class column, and outputs concatenated df'''
    
    df = pd.DataFrame(pd.read_csv('https://raw.githubusercontent.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking/master/data/joint/300w_infanface_train.csv'))
    df_adult = df.iloc[:3147, 1:]
    print(f"Shape of adult df: {df_adult.shape}")
    df_baby = df.iloc[3148:3357, 1:]
    print(f"Shape of baby df: {df_baby.shape}")
    num_adult_points = df_adult.shape[0]
    num_baby_points = df_baby.shape[0]
    total_points = num_adult_points + num_baby_points
    adult_p = round(num_adult_points/total_points,2)*100
    baby_p = round(num_baby_points/total_points,2)*100
    print(f"Total dataset is {adult_p}% adult points and {baby_p}% baby points")
    df_adult['class'] = 0
    df_baby['class'] = 1
    combine_df = pd.concat([df_adult,df_baby])
    
    return df_adult, df_baby, combine_df


def get_Xy(combine_df):
    '''
    Splits the data into X(data) and y(target)
    Focusing on the main coordinate columns (not the scale, w, h)
    '''
    X = combine_df.iloc[:, 4:-1]
    y = combine_df.iloc[:, -1:]
    X = X.to_numpy()
    y = y.to_numpy()
    return X,y

def main():
    df_adult, df_baby, combine_df = get_data()
    X,y = get_Xy(combine_df)
if __name__ == "__main__":
    main()