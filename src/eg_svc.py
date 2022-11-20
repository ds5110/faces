#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 23:17:59 2022

@author: jhautala
@author: Zongyu Wu
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# project
from util import meta_cache
from util.plot import scatter

# ----- constants, util functions and classes
pred = ['boxratio', 'interoc_norm', 'yaw_abs', 'roll_abs']
pred_plus = ['image_name', *pred, 'baby']
pcs = [f'pc{i}' for i in range(1, 5)]
savefig = False
fewer_plots = True


# ----- main execution
df = meta_cache.get_meta()[pred_plus]
X, y = df[pred], df['baby']

# ----- apply Zongyu's model
pca = PCA(whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
n_components = range(1, len(pred) + 1)

model = make_pipeline(pca, svc)
param_grid = {
    'svc__C': np.logspace(-3, 3, 7),
    'svc__gamma': np.logspace(-3, 3, 7),
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
    data=best_pca.fit_transform(X),
    columns=pcs,
    index=df.index,
)
comp_df = pd.concat([df, comp_df], axis=1)
comp_df['baby_hat'] = best_model.predict(X)

# ----- plot the first two principal components
for i in range(1, 5):
    for j in range(i + 1, 5):
        if fewer_plots and (i != 1 or j != 3): continue;
        scatter(
            f'SVC with RBF - PCA components {i} and {j} (of 4)',
            'PCA_{col1}_vs_{col2}_of_4.png',
            comp_df,
            [f'pc{i}', f'pc{j}'],
            'baby',
            target_name='Baby',
            alt_name='Adult',
            savefig=savefig,
        )

# ----- review the actual components
components = best_pca.get_feature_names_out()
weights = best_pca.components_
# good for general PCA, but I want column names to match the plots
# result = pd.DataFrame(weights.T, columns=components, index=X.columns)
result = pd.DataFrame(
    weights.T,
    columns=pcs,
    index=X.columns,
)
print(result)

# ------ plot predictor pairs
for i in range(4):
    for j in range(i + 1, 4):
        if fewer_plots and (i != 0 or j != 2): continue;
        scatter(
            'SVC with RBF - A few predictors',
            'scatter_{col1}_vs_{col2}_of_4.png',
            comp_df,
            [pred[i], pred[j]],
            'baby',
            target_name='Baby',
            alt_name='Adult',
            savefig=savefig,
        )
