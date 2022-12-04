"""
Trains simple Bayesian models, then Plots decision boundaries, confusion matrices, ROC curves and Detection Error Tradeoff curves for each classifier
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, DetCurveDisplay, RocCurveDisplay

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.inspection import DecisionBoundaryDisplay

def EDA(data):
    # quick scatter of two features that were called out in the paper,
    # Face propotion and interocular distance relative to face size

    ax = sns.scatterplot(data=data, x='boxratio', y='boxsize/interoc', hue='baby')
    ax.set(
        xlabel='Face Height / Width',
        ylabel='Face Size relative to Interocular Dist.',
        title='Exploring Table II'
    )

    leg = ax.axes.get_legend()
    leg.set_title('Age Group')
    legend_labels = ['Adult', 'Baby']
    for t, l in zip(leg.texts, legend_labels):
        t.set_text(l)

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

def plot_clf_boundary(clf, X, y, y_pred, ax, name, score):
    '''
    Only works with 2 features.
    Creates a scatter plot of all test points, 
    labeled as True and False, Adult and Baby,
    with the classifier's decision boundary
    '''
    
    plot_title = f'{name} \n score = {score:.4f}'

    ax.set(
    xlabel='Face Height / Width',
    ylabel='Face Size relative to Interocular Dist.',
    title= plot_title
    )

    tp = y == y_pred  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: Adults
    ax.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="red", label='True Adult')
    ax.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="#990000", label='False Adult')  # dark red

    # class 1: Babies
    ax.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="blue", label='True Baby')
    ax.scatter(X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="#000099", label='False Baby')  # dark blue

    disp = DecisionBoundaryDisplay.from_estimator(
        clf, 
        X, 
        response_method="predict", 
        xlabel='Face Height / Width', 
        ylabel='Face Size relative to Interoc Dist.', 
        alpha=0.5, 
        cmap='red_blue_classes', 
        eps=0.5, 
        ax=ax)

    ax.legend(title='Predictions')
    return ax
 
def train_plot(X, y, title):
    # split 50% data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.66, random_state=42)
    
    # normalized data
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    knn_params = {'n_neighbors': range(5,100, 5), 'weights': ['uniform', 'distance']}
       
    # create all models
    classifiers = {
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver="svd", store_covariance=True),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(store_covariance=True),
        "Gaussian Naive Bayes": GaussianNB(),
        "K Nearest Neighbors": GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, scoring='roc_auc')
    }

    # figure for ROC and DET curves
    fig_metrics, [ax_roc, ax_det] = plt.subplots(1, 2, figsize = (12,6))
    # figure grid of confusion matrices for each estimator
    fig_confusion_matrices, axs_confusion_matrices = plt.subplots(2, 2, figsize = (6, 6))
    figs = [fig_metrics, fig_confusion_matrices]

    # axes arrays need to be flattened 
    axs_confusion_matrices = axs_confusion_matrices.flatten()

    # if we have 2 features then we can plot the decision boundary
    if X_test.shape[1] == 2:
            fig_boundaries, axs_boundaries = plt.subplots(2, 2, figsize = (10, 10))
            figs.append(fig_boundaries)
            axs_boundaries = axs_boundaries.flatten()

    # train and plot each classifier
    for i, (name, clf) in enumerate(classifiers.items()):
        y_predict = clf.fit(X_train, y_train).predict(X_test)
        score = clf.score(X_test, y_test)

        # plot ROC curve
        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
        
        # plot detection error tradeoff curve
        DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
        ax_det.legend(loc='best')

        # plot a confusion matrix for the classifier
        ConfusionMatrixDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            display_labels = ['Adult', 'Infant'],
            ax = axs_confusion_matrices[i],
            colorbar=False)
        axs_confusion_matrices[i].set(title = name)
        
        # if we have 2 features then we can plot the decision boundary
        if X_test.shape[1] == 2:
            plot_clf_boundary(clf, X_test, y_test, y_predict, axs_boundaries[i], name, score)
            
            # if using GridSearchCV, add a text box of the selected params instead of a legend
            if isinstance(clf, GridSearchCV):
                axs_boundaries[i].get_legend().remove()

                # get a dict of the params from the best GridSearchCV and write a string of each key = value
                best_params = clf.best_estimator_.get_params()
                best_params_label = ''
                for param, value in best_params.items():
                    best_params_label += f'{param} = {value}\n'
                
                # create a text box in the upper left with the parameters
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                axs_boundaries[i].text(
                    x=0.65,
                    y=0.95,
                    s=best_params_label,
                    transform=axs_boundaries[i].transAxes,
                    fontsize=8, horizontalalignment='left',
                    verticalalignment='top',
                    bbox=props)
    
    for fig in figs:
        fig.suptitle(title, fontsize=18)
        fig.tight_layout()

def main():

    # get full dataset
    df = pd.read_csv('.\data\merged_landmarks.csv')
    
    # make the colormap
    make_cm()
    
    # first show plots using just 2 features
    title = '2 Features: \n boxratio, boxsize'
    X = df[['boxratio', 'boxsize/interoc']]
    y = df[['baby']].squeeze()
    train_plot(X, y, title)

    '''# run the same models with all 68 landmarks
    title = '136 Features: \n x,y for 68 landmarks'
    X = df.loc[:, df.columns.str.startswith('norm_cenrot')]
    y = df[['baby']].squeeze()
    train_plot(X, y, title)'''
    
    plt.show()

if __name__ == '__main__':
    main()