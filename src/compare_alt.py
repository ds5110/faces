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

# project
from util import cache, alt

pred = ['boxratio', 'interoc_norm', 'yaw_abs', 'roll_abs']


def get_some_cols(cache, cols):
    df = cache.get_meta('decorated')
    df = df[cols]
    return df


def get_Xy(df):
    return df[pred], df['baby']


# get decorated DFs
df_baby = get_some_cols(cache, ['image_name', *pred])
df_adult = get_some_cols(alt, ['image_name', *pred])

# add column for
df_baby['baby'] = 1
df_adult['baby'] = 0
df = pd.concat([df_baby, df_adult])
X, y = get_Xy(df)

# apply Zongyu's model
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
