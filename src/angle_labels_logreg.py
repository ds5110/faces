#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:50:42 2022

@author: jhautala
"""

from numpy.random import random_sample
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# project
from util import meta_cache


def eg_logreg(df, pred, target):
    X = df[pred]
    y = df[target]
    print(
        f'looking at "{target}" per {pred} '
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
    dummy.fit(X_train, y_train)
    dummy_y_hat = dummy.predict(X_test)
    dummy_score = accuracy_score(y_test, dummy_y_hat)
    print(f'\tdummy score: {dummy_score:.3f}')


df = meta_cache.get_meta('baby')

# test logistic regression with random predictor
# tmp = df.copy()
# tmp['r'] = random_sample((tmp.shape[0],))
# for t in ['tilted','turned']:
#     test_logreg(tmp,['r'],t)

for t in ['tilted', 'turned']:
    for pp in [['yaw_abs'], ['roll_abs'], ['yaw_abs', 'roll_abs']]:
        eg_logreg(df, pp, t)
