"""
Models Linear Discriminant Analysis, Quadratic Discriminant Analysis and Guassian Naive Bayes
Plots decision boundaries, ROC curves and Detection Error Tradeoff curves for each classifier
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, DetCurveDisplay, RocCurveDisplay

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

"""
Creates a scatter plot of all test points, 
labeled as True and False, Adult and Baby,
with the classifier's decision boundary
"""
def plot_boundary(clf, X, y, y_pred, ax, name):
    title= name + ' on Table 2 Features'
    ax.set(
    xlabel='Face Height / Width',
    ylabel='Face Size relative to Interocular Dist.',
    title= title
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
    
def main():
    # get full dataset
    df = pd.read_csv('.\data\merged_landmarks.csv')
    make_cm()

    # set up data for model training and testing
    scalar = StandardScaler()
    X = df[['boxratio', 'boxsize/interoc']]
    X = scalar.fit_transform(X)
    y = df[['baby']].squeeze()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # create all three models
    classifiers = {
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(solver="svd", store_covariance=True),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(store_covariance=True),
        "Gaussian Naive Bayes": GaussianNB()
    }

    # Train and plot boundaries
    fig_metrics, [ax_roc, ax_det] = plt.subplots(1,2,figsize=(12,6))
    fig_boundaries, axs_b = plt.subplots(1,3,figsize=(18,6))

    for i, (name, clf) in enumerate(classifiers.items()):
        y_predict = clf.fit(X_train, y_train).predict(X_test)
        RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
        DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)
        boundary = plot_boundary(clf, X_test, y_test, y_predict, axs_b[i], name)

    plt.show()

if __name__ == '__main__':
    main()