"""
@author: jhautala
"""

import numpy as np
from numpy.random import random_sample
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# project
from util import meta_cache
from util.plot import scatter

savefig = True

df = meta_cache.get_meta()
df['roll_x_boxratio'] = df['roll_abs'] * df['boxratio']
df['roll_x'] = np.cbrt(df['roll_abs'] * df['boxratio'] * df['interoc_norm'])

# ------ setup a bunch of column groups
sz_cols = ['width', 'height', 'cenrot-width', 'cenrot-height']
meta_cols = ['image-set', 'filename', 'partition', 'subpartition']

# coordinates
cenrot = [col for col in df.columns if col.startswith('cenrot-')]
norm = [col for col in df.columns if col.startswith('norm-')]
norm_cenrot = [col for col in df.columns if col.startswith('norm_cenrot-')]

# diff cols
sym_diff = [col for col in df.columns if col.startswith('sym_diff-')]
cenrot_sym_diff = [col for col in df.columns if col.startswith('cenrot_sym_diff-')]
norm_cenrot_sym_diff = [col for col in df.columns if col.startswith('norm_cenrot_sym_diff-')]

# higher-level cols
zongyu_pred = ['boxratio', 'interoc_norm']
angle_pred = ['yaw', 'roll']
abs_angle_pred = ['yaw_abs', 'roll_abs', ]
pred = [*zongyu_pred, *angle_pred]
pred_abs = [*zongyu_pred, *abs_angle_pred]

# presumed useful
more_pred = [*norm_cenrot_sym_diff, *norm_cenrot]

target = 'baby'


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
    pipe = Pipeline(
        steps=[
            ('poly', PolynomialFeatures(degree=3)),
            ('logreg', LogisticRegression()),
        ],
    )
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)
    score = accuracy_score(y_test, y_hat)
    print(
        f'\tlogreg score: '
        f'{score:.3f}')
    
    # print('coefficients:')
    # logreg = pipe.named_steps['logreg']
    # for (p, c) in zip(logreg.feature_names_in_, logreg.coef_[0]):
    #     print(f'\t\t{p}: {c}')

    # fit uniform dummy, to compare
    # NOTE: we don't really need to use train/test
    #       unless we use 'stratified' strategy
    dummy = DummyClassifier(strategy='uniform')
    dummy.fit(X_train, y_train)
    dummy_y_hat = dummy.predict(X_test)
    dummy_score = accuracy_score(y_test, dummy_y_hat)
    print(f'\tdummy score: {dummy_score:.3f}')
    
    df[f'{target}_hat'] = pipe.predict(X)
    return df, pipe, score


# test logistic regression with random predictor
# tmp = df.copy()
# tmp['r'] = random_sample((tmp.shape[0],))
# for t in ['tilted','turned']:
#     test_logreg(tmp,['r'],t)

for pp in [
    pred_abs,
    # ['roll_x_boxratio', *pred_abs],
    # ['roll_x', *pred_abs]
]:
    tmp, model, score = eg_logreg(df, pp, 'baby')
    scatter(
        f'Logistic Regression - baby vs {", ".join(pp)} \n '
            f'score: {score:.3f}',
        f'scatter_boxratio_vs_yaw.png',
        tmp,
        ['boxratio', 'yaw_abs'],
        target,
        target_name='Baby',
        alt_name='Adult',
        savefig=savefig
    )
