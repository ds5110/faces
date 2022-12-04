"""
Trains models, then plots  confusion matrices, ROC curves and Detection Error Tradeoff curves for each classifier.
If the model is 2-dimensional, also plots decision boundaries
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

def train(X, y, classifiers):
    # split 50% data for testing
    X_train_scaled, X_test_scaled, y_train, y_test= train_test_split(X, y, test_size=0.66, random_state=42)
    
    # fit scalar to training data and normalize data
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train_scaled)
    X_test_scaled = scalar.transform(X_test_scaled)

    trained_classifiers = {}
    scores = {}

    for name, classifier in classifiers.items():
        trained_classifiers[name] = classifier.fit(X_train_scaled, y_train)
        # if using GridSearchCV, add a text box of the selected params instead of a legend
        scores[name] = classifier.score(X_test_scaled, y_test)

    return trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test

def plot_metrics(trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test, fig_title):
    '''
    Create 2 figures showing metrics for trained classifiers:
    * One figure with confusion matrices for each classifier
    * One figure with ROC Curves and DET curves for all classifiers 
    '''
    
    fig_metrics, [ax_roc, ax_det] = plt.subplots(1, 2, figsize = (12,6))
    fig_confusion_matrices, axs_confusion_matrices = plt.subplots(2, 2, figsize = (6, 6))
    figs = [fig_metrics, fig_confusion_matrices]

    # axes arrays need to be 1-dimensional to index through them in the next loop
    axs_confusion_matrices = axs_confusion_matrices.flatten()

    for i, (name, trained_classifier) in enumerate(trained_classifiers.items()):
        ax_title = f'{name} \n score = {scores[name]:.4f}'
        
        # plot ROC curve
        RocCurveDisplay.from_estimator(trained_classifier, X_test_scaled, y_test, ax=ax_roc, name=name)
        
        # plot detection error tradeoff curve
        DetCurveDisplay.from_estimator(trained_classifier, X_test_scaled, y_test, ax=ax_det, name=name)
        ax_det.legend(loc='best')

        # plot a confusion matrix for the classifier
        ConfusionMatrixDisplay.from_estimator(
            trained_classifier,
            X_test_scaled,
            y_test,
            display_labels = ['Adult', 'Infant'],
            ax = axs_confusion_matrices[i],
            colorbar=False)
        axs_confusion_matrices[i].set(title = ax_title)
                
    for fig in figs:
        fig.suptitle(fig_title, fontsize=18)
        fig.tight_layout()

def plot_clf_boundary(clf, X_test_scaled, y_test, y_pred, ax, ax_title):
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
    X0, X1 = X_test_scaled[y_test == 0], X_test_scaled[y_test == 1]
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
        X_test_scaled, 
        response_method="predict", 
        xlabel='Face Height / Width', 
        ylabel='Face Size relative to Interoc Dist.', 
        alpha=0.5, 
        cmap='red_blue_classes', 
        eps=0.5, 
        ax=ax)

    ax.legend(title='Predictions')
    return ax

def plot2D(trained_classifiers, scores, X_test_scaled, y_test, fig_title):
    '''
    Loops through each model and plots their boundary using plot_clf_boundary()
    '''
    assert X_test_scaled.shape[1] == 2    
    fig_boundaries, axs_boundaries = plt.subplots(2, 2, figsize = (10, 10))
    axs_boundaries = axs_boundaries.flatten()

    for i, (name, trained_classifier) in enumerate(trained_classifiers.items()):
        ax_title = f'{name} \n score = {scores[name]:.4f}'
        y_predict = trained_classifier.predict(X_test_scaled)
        plot_clf_boundary(trained_classifier, X_test_scaled, y_test, y_predict, axs_boundaries[i], ax_title)
        
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
    best_params_message = ''
    for param, value in best_params.items():
        best_params_message += f'{param} = {value}\n'
    return best_params_message

def main():
    # get full dataset
    df = pd.read_csv('.\data\merged_landmarks.csv')
    
    # make the colormap
    make_cm()

    # create all models
    knn_params = {'n_neighbors': range(5,100, 5), 'weights': ['uniform', 'distance']}
    classifiers = {
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver="svd", store_covariance=True),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(store_covariance=True),
        "Gaussian Naive Bayes": GaussianNB(),
        "K Nearest Neighbors": GridSearchCV(estimator=KNeighborsClassifier(), param_grid=knn_params, scoring='roc_auc')
    }
    
    # using just 2 features
    fig_title = '2 Features: \n boxratio, boxsize'
    X = df[['boxratio', 'boxsize/interoc']]
    y = df[['baby']].squeeze()
    trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test = train(X, y, classifiers)
    plot_metrics(trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test, fig_title)
    plot2D(trained_classifiers, scores, X_test_scaled, y_test, fig_title)

    # using all 68 original landmarks
    fig_title = '136 Features: \n Original 68 landmarks'
    X = df.loc[:, df.columns.str.startswith('x') + df.columns.str.startswith('y')]
    y = df[['baby']].squeeze()
    trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test = train(X, y, classifiers)
    plot_metrics(trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test, fig_title)

    # using all 68 normalized and centered landmarks
    fig_title = '136 Features: \n Normalized, Centered 68 landmarks'
    X = df.loc[:, df.columns.str.startswith('norm_cenrot')]
    y = df[['baby']].squeeze()
    trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test = train(X, y, classifiers)
    plot_metrics(trained_classifiers, scores, X_train_scaled, X_test_scaled, y_train, y_test, fig_title)
    
    plt.show()

if __name__ == '__main__':
    main()