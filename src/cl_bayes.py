"""
Trains models, then plots  confusion matrices, ROC curves and Detection Error Tradeoff curves for each classifier.
If the model is 2-dimensional, also plots decision boundaries
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, DetCurveDisplay, RocCurveDisplay

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

def make_cm():        
    # create a cmap for decision boundary plots
    from matplotlib import colors
    cmap = colors.LinearSegmentedColormap(
        "red_blue_classes",
        {
            "red": [(0, 1, 1), (1, 0.7, 0.7)],
            "green": [(0, 0.7, 0.7), (1, 0.7, 0.7)],
            "blue": [(0, 0.7, 0.7), (1, 1, 1)],
        },
    )
    plt.cm.register_cmap(cmap=cmap)

def train(X_train, X_test, y_train, y_test, classifiers):
    
    test_scores = {}
    train_scores = {}

    for name, classifier in classifiers.items():
        
        classifier.fit(X_train, y_train)
        
        # GridSearchCV requires getting the best estimator via attribute 
        if isinstance(classifier, GridSearchCV):
            train_scores[name] = classifier.best_score_ # Mean cross-validated score of the best_estimator
            test_scores[name] = classifier.best_estimator_.score(X_test, y_test)
        else:
            train_scores[name] = classifier.score(X_train, y_train)
            test_scores[name] = classifier.score(X_test, y_test)

    return classifiers, test_scores, train_scores

def plot_metrics(trained_classifiers, test_scores, train_scores, X_test, y_test, fig_title):
    '''
    Create 2 figures showing metrics for trained classifiers:
    * One figure with confusion matrices for each classifier
    * One figure with ROC Curves and DET curves for all classifiers 
    '''
    
    fig_metrics, [ax_roc, ax_det] = plt.subplots(1, 2, figsize = (12,6))
    fig_confusion_matrices, axs_confusion_matrices = plt.subplots(2, 2, figsize = (8, 6))
    figs = [fig_metrics, fig_confusion_matrices]

    # axes arrays need to be 1-dimensional to index through them in the next loop
    axs_confusion_matrices = axs_confusion_matrices.flatten()

    for i, (name, trained_classifier) in enumerate(trained_classifiers.items()):
        ax_title = f'{name} \ntrain score = {train_scores[name]:.4f} \ntest score = {test_scores[name]:.4f}'
        
        # GridSearchCV requires getting the best estimator via attribute 
        if isinstance(trained_classifier, GridSearchCV):
            chosen_classifier = trained_classifier.best_estimator_
        else:
            chosen_classifier = trained_classifier
        
        # plot ROC curve
        RocCurveDisplay.from_estimator(chosen_classifier, X_test, y_test, ax=ax_roc, name=name)
        
        # plot detection error tradeoff curve
        DetCurveDisplay.from_estimator(chosen_classifier, X_test, y_test, ax=ax_det, name=name)
        ax_det.legend(loc='best')

        # plot a confusion matrix for the classifier
        ConfusionMatrixDisplay.from_estimator(
            chosen_classifier,
            X_test,
            y_test,
            display_labels = ['Adult', 'Infant'],
            ax = axs_confusion_matrices[i],
            colorbar=False)
        axs_confusion_matrices[i].set(title = ax_title)
                
    for fig in figs:
        fig.suptitle(fig_title, fontsize=18)
        fig.tight_layout()

def plot_clf_boundary(clf, X_test, y_test, y_pred, ax, ax_title):
    '''
    Creates a scatter plot of all test points labeled as True and False, Adult and Baby,
    and plots the classifier's decision boundary. Only works with 2d models.'''

    ax.set(
    xlabel='Face Height / Width',
    ylabel='Face Size relative to Interocular Dist.',
    title= ax_title
    )

    tp = y_test == y_pred  # True Positive
    tp0, tp1 = tp[y_test == 0], tp[y_test == 1]
    X0, X1 = X_test[y_test == 0], X_test[y_test == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: Adults
    ax.scatter(X0_tp.iloc[:, 0], X0_tp.iloc[:, 1], marker=".", color="red", label='True Adult')
    ax.scatter(X0_fp.iloc[:, 0], X0_fp.iloc[:, 1], marker="x", s=20, color="#990000", label='False Adult')  # dark red

    # class 1: Babies
    ax.scatter(X1_tp.iloc[:, 0], X1_tp.iloc[:, 1], marker=".", color="blue", label='True Baby')
    ax.scatter(X1_fp.iloc[:, 0], X1_fp.iloc[:, 1], marker="x", s=20, color="#000099", label='False Baby')  # dark blue

    disp = DecisionBoundaryDisplay.from_estimator(
        clf, 
        X_test, 
        response_method="predict", 
        xlabel='Face Height / Width', 
        ylabel='Face Size relative to Interoc Dist.', 
        alpha=0.5, 
        cmap='red_blue_classes', 
        eps=0.5, 
        ax=ax)

    ax.legend(title='Predictions')
    return ax

def plot2D(trained_classifiers, test_scores, X_test, y_test, fig_title):
    '''
    Loops through each model and plots their boundary using plot_clf_boundary()
    '''
    assert X_test.shape[1] == 2    
    fig_boundaries, axs_boundaries = plt.subplots(2, 2, figsize = (10, 10))
    axs_boundaries = axs_boundaries.flatten()

    for i, (name, trained_classifier) in enumerate(trained_classifiers.items()):
        ax_title = f'{name} \ntest score = {test_scores[name]:.4f}'
        y_predict = trained_classifier.predict(X_test)
        plot_clf_boundary(trained_classifier, X_test, y_test, y_predict, axs_boundaries[i], ax_title)
        
        # if GridCV was used, then add the chosen parameters to the plot
        if isinstance(trained_classifier, GridSearchCV):
            axs_boundaries[i].get_legend().remove()
            # create a text box in the upper left with the parameters
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            best_params_message = GridSearchCV_Parameter_Message(trained_classifier)
            axs_boundaries[i].text(
                x=0.65,
                y=0.95,
                s=best_params_message,
                transform=axs_boundaries[i].transAxes,
                fontsize=8, horizontalalignment='left',
                verticalalignment='top',
                bbox=props)
    
    fig_boundaries.suptitle(fig_title, fontsize=18)
    fig_boundaries.tight_layout()

def GridSearchCV_Parameter_Message(trained_classifier):
    assert isinstance(trained_classifier, GridSearchCV) 
    # get a dict of the params from the best GridSearchCV and write a string of each key = value
    best_params = trained_classifier.best_estimator_.get_params()
    # get the name of the classifier step in the pipeline
    classifier_step_name = best_params['steps'][1][0] + '__'
    best_params_message = ''
    for param, value in best_params.items():
        if param.startswith(classifier_step_name):
            param_name = param.split('__',1)[1]
            best_params_message += f'{param_name} = {value}\n'
    return best_params_message

def train_and_plot(X,y, classifiers, fig_title):
    # hold out data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # train the classifiers
    trained_classifiers, test_scores, train_scores = train(X_train, X_test, y_train, y_test, classifiers)

    # plot the results    
    plot_metrics(trained_classifiers, test_scores, train_scores, X_test, y_test, fig_title)
    if X_test.shape[1] == 2:
        plot2D(trained_classifiers, test_scores, X_test, y_test, fig_title)

def main():
    # get full dataset
    df = pd.read_csv('.\data\merged_landmarks.csv')
    
    # make the colormap
    make_cm()

    # create all models
    lda_params = {'lda__solver': ['svd', 'lsqr', 'eigen'], 'lda__shrinkage': [None, 'auto']}
    qda_params = {'qda__reg_param': np.arange(0, 1, 0.05)}
    knn_params = {'knn__n_neighbors': range(5,100, 5), 'knn__weights': ['uniform', 'distance']}
    classifiers = {
        "Linear Discriminant Analysis": GridSearchCV(
            estimator = Pipeline([
            ('scaler', StandardScaler()), 
            ('lda', LinearDiscriminantAnalysis(store_covariance=True, tol=.01))]),
            param_grid = lda_params, 
            scoring = 'roc_auc'),
        
        "Quadratic Discriminant Analysis": GridSearchCV(
            estimator = Pipeline([
            ('scaler', StandardScaler()), 
            ('qda', QuadraticDiscriminantAnalysis(store_covariance=True))]),
            param_grid = qda_params, 
            scoring = 'roc_auc'),
        
        "Gaussian Naive Bayes":
            Pipeline([
            ('scaler', StandardScaler()),
            ('gnb',GaussianNB())]),
        
        "K Nearest Neighbors": GridSearchCV(
            estimator = Pipeline([
                ('scaler', StandardScaler()), 
                ('knn', KNeighborsClassifier())]), 
            param_grid = knn_params, 
            scoring = 'roc_auc')
    }
    
    # using just 2 features
    fig_title = '2 Features: \n boxratio, boxsize'
    X = df[['boxratio', 'boxsize/interoc']]
    y = df[['baby']].squeeze()
    train_and_plot(X, y, classifiers, fig_title)

    # using all 68 original landmarks
    fig_title = '136 Features: \n Original 68 landmarks'
    X = df.loc[:, df.columns.str.startswith('x') + df.columns.str.startswith('y')]
    y = df[['baby']].squeeze()
    train_and_plot(X, y, classifiers, fig_title)

    # using all 68 normalized and centered landmarks
    fig_title = '136 Features: \n Normalized, Centered 68 landmarks'
    X = df.loc[:, df.columns.str.startswith('norm_cenrot')]
    y = df[['baby']].squeeze()
    train_and_plot(X, y, classifiers, fig_title)

    plt.show()

if __name__ == '__main__':
    main()