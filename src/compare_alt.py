#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:17:59 2022

@author: jhautala
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# project
from util import cache, alt


#------ constants, util functions and classes
pred = ['boxratio', 'interoc_norm', 'yaw_abs', 'roll_abs']


def get_some_cols(cache, cols):
    df = cache.get_meta('decorated')
    df = df[cols]
    return df


def get_Xy(df):
    return df[pred], df['baby']


class Case:
    def __init__(self,name,color,mask):
        self.name = name
        self.color = color
        self.mask = mask


#------ get the data together
# get decorated DFs
df_baby = get_some_cols(cache, ['image_name', *pred])
df_adult = get_some_cols(alt, ['image_name', *pred])

# add column for
df_baby['baby'] = 1
df_adult['baby'] = 0
df = pd.concat([df_baby, df_adult])
X, y = get_Xy(df)

#------ apply Zongyu's model
pca = PCA(whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
n_components = range(1, len(pred) + 1)

model = make_pipeline(pca, svc)

param_grid = {
    'svc__C': [1, 5, 10, 50],
    'svc__gamma': [0.0001, 0.0005, 0.001, 0.005],
    'pca__n_components': n_components,
}

grid = GridSearchCV(model, param_grid, return_train_score=True, n_jobs=5)

grid.fit(X, y)

print(grid.best_params_)
print(f"best score: {grid.best_score_:.3f}")

best_model = grid.best_estimator_
best_pca = best_model.named_steps['pca']
best_svc = best_model.named_steps['svc']

comp_df = pd.DataFrame(
    data = best_pca.fit_transform(X),
    columns = [f'pc{i}' for i in range(1,5)],
    index=df.index,
)
comp_df = pd.concat([df,comp_df], axis=1)
comp_df['baby_hat'] = best_model.predict(X)


#------ review the actual components
components = best_pca.get_feature_names_out()
weights = best_pca.components_
result = pd.DataFrame(weights.T, columns=components, index=X.columns)
print(result)

#------ plot the first two principal components
def plot_components(i,j):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(f'Principal Component {i}', fontsize = 15)
    ax.set_ylabel(f'Principal Component {j}', fontsize = 15)
    ax.set_title(f'PCA - Components {i} and {j} (of 4)', fontsize = 20)
    
    t = comp_df['baby' ] == 1
    f = comp_df['baby' ] == 0
    p = comp_df['baby_hat' ] == 1
    n = comp_df['baby_hat' ] == 0
    cases = [
        Case('True Adult', 'tab:green', f & n),
        Case('True Baby', 'tab:blue', t & p),
        Case('False Adult', 'tab:red', t & n),
        Case('False Baby', 'tab:orange', f & p),
    ]
    for case in cases:
        ax.scatter(
            comp_df[case.mask][f'pc{i}'],
            comp_df[case.mask][f'pc{j}'],
            c = case.color,
            s = 50,
            label=case.name
        )
    ax.legend()
    plt.savefig(
        f'figs/PCA_{i}_and_{j}_of_4.png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.show()

for i in range(1,5):
    for j in range(i+1,5):
        plot_components(i,j)