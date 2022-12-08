"""
@author: jhautala
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import random_sample
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import confusion_matrix

# project
from util import meta_cache
from util.plot import scatter
from util.column_names import norm_cenrot_sym_diff, norm_cenrot_cols

save_fig = False

df = meta_cache.get_meta()
df['roll_x_boxratio'] = df['roll_abs'] * df['boxratio']
df['roll_x'] = np.cbrt(df['roll_abs'] * df['boxratio'] * df['interoc_norm'])

# higher-level cols
zongyu_pred = ['boxratio', 'interoc_norm']
angle_pred = ['yaw', 'roll']
abs_angle_pred = ['yaw_abs', 'roll_abs', ]
pred = [*zongyu_pred, *angle_pred]
pred_abs = [*zongyu_pred, *abs_angle_pred]

# presumed useful
more_pred = [*norm_cenrot_sym_diff, *norm_cenrot_cols]

target = 'baby'

def eg_logreg(df, pred, target, poly=False):
    X = df[pred]
    y = df[target]
    print(
        f'looking at \'{target}\' per {pred} '
        f'(n: {X.shape[0]}; p: {X.shape[1]})'
    )

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        random_state=42,
    )

    # fit logreg
    steps = []
    if poly:
        steps.append(('poly', PolynomialFeatures(degree=3)))
    steps.append(('logreg', LogisticRegression()))
    pipe = Pipeline(
        steps=steps,
    )
    pipe.fit(X_train, y_train)

    y_hat = pipe.predict(X_test)

    # - classification report
    cls_rpt = classification_report(
        y_test,
        y_hat,
        target_names=['baby', 'adult'],
    )
    width = len(cls_rpt.split('\n')[0])
    line = '='*width
    print('\n' + '\n'.join([line, 'Classification Report', line]))
    print(cls_rpt)

    # accuracy score
    score = accuracy_score(y_test, y_hat)
    print(
        f'\tlogreg score: '
        f'{score:.3f}')

    # TODO: figure out how to print coefficients when using poly features...
    if not poly:
        print('\tcoefficients:')
        logreg = pipe.named_steps['logreg']
        # NOTE: it is not relevant to sort these here, but I couldn't help it /shrug
        coefs = np.array([[p, c] for (p, c) in zip(logreg.feature_names_in_, logreg.coef_[0])])
        ii = np.flip(np.argsort(coefs[:,1], axis=0))
        for (pred, coef) in [(coefs[i,0], coefs[i,1]) for i in ii]:
            print(f'\t\t{pred}: {coef}')

    # fit uniform dummy, to compare
    # NOTE: we don't really need to use train/test
    #       unless we use 'stratified' strategy
    dummy = DummyClassifier(strategy='stratified', random_state=42)
    dummy.fit(X_train, y_train)
    dummy_y_hat = dummy.predict(X_test)
    dummy_score = accuracy_score(y_test, dummy_y_hat)
    print(f'\tdummy score: {dummy_score:.3f}')
    
    df[f'{target}_hat'] = pipe.predict(X)
    return df, pipe, score, X_test, y_test, y_hat


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
    tmp, model, score, X_test, y_test, y_hat = eg_logreg(df, pp, 'baby')
    test_df = pd.DataFrame(X_test, columns=pp)
    test_df[target] = y_test
    test_df[f'{target}_hat'] = y_hat
    scatter_cols = ['boxratio', 'yaw_abs']
    scatter(
        f'Logistic Regression - baby vs {", ".join(pp)} \n '
            f'score: {score:.3f}',
        f'scatter_{" vs ".join(scatter_cols)}.png',
        tmp, # test_df,
        scatter_cols,
        target,
        target_name='Baby',
        alt_name='Adult',
        save_fig=save_fig
    )

    def conf_mat(desc, y, y_hat):
        target_names = ['Baby', 'Adult']
        mat = confusion_matrix(y, y_hat)
        sns.heatmap(
            mat.T,
            square=True,
            annot=True,
            fmt='d',
            cbar=False,
            xticklabels=target_names,
            yticklabels=target_names,
        )
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.title(f'Confusion Matrix for {desc} Results')
        plt.tight_layout()
        if save_fig:
            plt.savefig(
                f'figs/logreg_4_pred_conf_mat_{desc}.png',
                dpi=300,
                bbox_inches='tight',
            )
        plt.show()

    conf_mat('Validation', y_test, y_hat)
    conf_mat('Logistic Regression', df[target], model.predict(df[pp]))