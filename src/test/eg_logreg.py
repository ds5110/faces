#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:50:42 2022

@author: jhautala
"""

import math
import pandas as pd
import numpy as np
from numpy.random import random_sample
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# project
from util import cache
from util.model import cat_cols

df = cache.get_meta('decorated')


#------ setup a bunch of column groups
sz_cols = ['width','height','cenrot-width','cenrot-height']
meta_cols = ['image-set','filename','partition','subpartition']

# coordinates
cenrot = [col for col in df.columns if col.startswith('cenrot-')]
norm = [col for col in df.columns if col.startswith('norm-')]
norm_cenrot = [col for col in df.columns if col.startswith('norm_cenrot-')]

# diff cols
sym_diff = [col for col in df.columns if col.startswith('sym_diff-')]
cenrot_sym_diff = [col for col in df.columns if col.startswith('cenrot_sym_diff-')]
norm_cenrot_sym_diff = [col for col in df.columns if col.startswith('norm_cenrot_sym_diff-')]

# higher-level cols
summ_cols = ['yaw','roll']
summ_cols_abs = ['yaw_abs','roll_abs']
all_summ = [*summ_cols,*summ_cols_abs]

# presumed useful
subset_pred = [*summ_cols,*norm_cenrot_sym_diff,*norm_cenrot]

# everything
all_pred = df.drop(columns=[*cat_cols,*meta_cols]).columns.values

# maybe less useful
other_pred = sorted(list(set(all_pred) - set(subset_pred)))

# def test_dummy(df,target):
#     clf = DummyClassifier(strategy='uniform')
#     X = np.zeros(df.shape[0])
#     y = df[target]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#     )
#     clf.fit(X_train,y_train)
#     y_hat = clf.predict(X_test)
#     score = accuracy_score(y_test, y_hat)
#     print(f'dummy score: {score:.3f}')

# dd = np.concatenate([np.zeros(100),np.ones(900)])
# tmp = pd.DataFrame(dd,columns=['t'])
# tmp['r'] = random_sample((1000,))
# test_dummy(tmp,'t')

def eg_logreg(df,pred,target):
    X = df[pred]
    y = df[target]
    print(
        f'looking at "{target} "per {pred} '
        f'(n: {X.shape[0]}; p: {X.shape[1]})'
    )
    
    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
    )
    
    # fit logreg
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    y_hat = logreg.predict(X_test)
    score = accuracy_score(y_test, y_hat)
    print(
        f'\tlogreg score: '
        f'{score:.3f}')
    
    # fit uniform dummy, to compare
    # NOTE: we don't really need to use train/test
    #       unless we use 'stratified' strategy
    dummy = DummyClassifier(strategy='uniform')
    dummy.fit(X_train,y_train)
    y_hat = dummy.predict(X_test)
    score = accuracy_score(y_test, y_hat)
    print(f'\tdummy score: {score:.3f}')

# test logistic regression with random predictor
# tmp = df.copy()
# tmp['r'] = random_sample((tmp.shape[0],))
# for t in ['tilted','turned']:
#     test_logreg(tmp,['r'],t)

for t in ['tilted','turned']:
    for pp in [['yaw_abs'],['roll_abs'],['yaw_abs','roll_abs']]:
        eg_logreg(df,pp,t)
