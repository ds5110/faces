#Sophia Cofone, 10/24/22, file is intended to explore the data, in particular the 4 boolean columns

import pandas as pd
import matplotlib.pyplot as plt
import itertools

'''Thought process behind this file:
Goal: I wanted to see some basic stats about the data, such as how many images are in the sets, and how many different images have the targets listed
Result: There are more images in the adds google and youtube sets, with less in the google2 and youtube2 sets (I think this makes sense as the 2 sets are for testing?)
I also found out that there are more images with just one of the targets, like just tilted or just expressive. less images are marked with multiple, and no images are marked with 'turned', 'occluded', and 'expressive'.
'''

path = './data/'
file_name = path + 'labels.csv'

#Creating dataframe
df = pd.DataFrame(pd.read_csv(file_name))

#simple plot to get an idea of "set" dist
def show_set(df):
    plt.hist(df['image-set'])
    plt.show()

show_set(df)


target_cols = ['turned', 'occluded', 'expressive', 'tilted']
dfnew = df[target_cols].copy()
nparray = dfnew.to_numpy()

#Trying to see which combinations of the targets come up
def see_combos(nparray):
    a = '10'
    combo_options = [''.join(output) for output in itertools.product(a,repeat=4)]
    combos = {}
    for i in range(len(combo_options)):
        combos[combo_options[i]]=0
    for i in range(nparray.shape[0]):
        result = ''.join([str(int(nparray[i][j])) for j in range(4)])
        if result in combos:
            combos[result]+=1
        else:
            print('very bade')
    return combos

#plot to get an idea of "combos" dist
def show_combos(nparray):
    plt.imshow(nparray, aspect='auto')
    plt.show()

print(see_combos(nparray))
show_combos(nparray)
