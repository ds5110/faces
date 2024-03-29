# DS5110 faces: LDA, QDA, Naive Bayes, K-Neighbors Summary

The purpose of this file is to explain the main takeaways from Bayesian and KNN model exploration.

### To reproduce the figures and models shown here:
run the following in your command prompt:
```
make bayes_figures
```

## 2-D Feature Space and Decision Boundaries

I think the best way to introduce these models is starting with a 'toy' version of the faces of the dataset; using only the `boxratio` and `boxsize/interoc` that are mentioned shown Table II in the InfAnFace paper:

<img src="figs/table2.png" width=400>

The creation of these features from the original landmarks is described in `preprocessing.md`. 

<img src="figs/bayes_eda.png" width=600>

Based on training models with just these two features and many techniques the irreducible error seems to be ~0.11, for predicting whether the original landmarks come from the helen dataset or InfAnFace dataset.

<img src="figs/bayes_2feature_boundary.png" width=800>

Titles of each plot show the modeling technique, and the accuracy score on a train-test-split of 40% test data.

The color boundary in each plot is the decision boundary, where the model places a 50% chance that an observation could be an infant or adult.

3 models - Linear Discriminant Analysis, Quadratic Discriminant Analysis, and Gaussian Naive Bayes - are Bayesian models.  

These models start by calculating a prior probability of each class based on their proportion in the data, i.e., 40 infant observations and 60 adult observations would result in prior probabilities of *P(infant)* = 0.40 and *P(adult)* = 0.60.  

Then, the models create posterior probabilities by updating those prior probabilities with calculations of each of the classes' distributions amongst the features.

The differences between the 3 models come from how each assumes different features are correlated with each class.

The simplest is Naive Bayes (GaussianNB), which assumes that all features are independent and normally distributed. While those assumptions are usually false and can lead to underperformance, this gives a low-variance technique that works well on high-dimensionality data. 

Linear Discriminant Analysis (LDA) still assumes a normal distribution for each feature, but does not assume that features are independent. The correlations between feature combinations is stored in a covariance matrix, that influences the posterior LDA.  

LDA calculates this covariance matrix assuming that the correlations are the same for every class. This reduces model complexity greatly, since the covariance matrix will be n*n for n features, leading to n<sup>2</sup> parameters for each matrix. Often, this strikes a useful bias-variance balance for typical datasets. 

Quadratic Discriminant Analysis (QDA) is the most complex and high-variance of the three, because it does the same calculations as LDA, but with a different covariance matrix for each class. 

Included in these figures is also a K-Nearest Neighbors classifier, that will be discussed in a later section.

So which model succeeds on the 2-D dataset?

<img src="figs/bayes_2feature_confusion_matrix.png" width=600>

These confusion matrices show how each classifier's accuracy varies for the adult and infant classes. All three coincidentally have the exact same accuracy score, but the recall score of infants is highest with LDA and about the same for QDA and GaussianNB.

I find it interesting that the decision boundaries for the simplest (GaussianNB) and most complex (QDA) are almost identical. However, the more useful distinction is seen on the training score, where QDA scores the highest.  

The caveat with a higher-variance model like QDA is that it will quickly overfit the data.

Here are ROC and Detection Error Tradeoff Curves for each model:

<img src="figs/bayes_2feature_metrics.png" width=600>

Area Under Curve (AUC) is a robust metric for comparing classifiers, where a higher score shows the model better avoids misclassification across all classification thresholds.

To show this same behavior on another feature selection:

<img src="figs/bayes_boxratio_vs_interoc_norm.png" width=400>
<img src="figs/bayes_interoc_norm_boundary.png" width=1000>
<img src="figs/bayes_interoc_norm_confusion_matrix.png" width=600>
<img src="figs/bayes_interoc_norm_metrics.png" width=600>

## The Ground Truth 68 Landmarks
Next, we're looking at the original 'ground truth' cooridinates in the InfAnFace dataset. These are x, y pixel coordinates for 68 labeled coordinates that were determined by human researchers. 

These have not been normalized in scale, so larger images will lead to different coordinates, regardless of the class label, and I expected that this noisy extrinsic variable would lead to poorer model test performance. 

<img src="figs/bayes_original_confusion_matrix.png" width=600>
<img src="figs/bayes_original_metrics.png" width=600>

Surprisingly, both LDA and QDA were able to achieve high test scores. I interpret this result as both and especially QDA being able to perform their own 'normalization' via a covariance matrix.  
Gaussian Niave Bayes performs poorly, and shows that without the covariance estimates, the noise from pixel coordinates overpower the model.  

KNN performs poorly as well. This can be expected, because KNN becomes highly sensitive to noisy features in high dimensional data.

## The Normalized, Centered and Rotated 68 Landmarks
Finally, we feed the models normalized, centered and rotated coordinates that we created, and see if removing pixel-base externalities improved the models' predictive capability.

<img src="figs/bayes_norm_confusion_matrix.png" width=600>
<img src="figs/bayes_norm_metrics.png" width=600>

GaussianNB is still the worst performer, but with 1/3 the error rate as with the original training set.

Whereas QDA was the best performer on the raw coordinates, it shows overfitting on the normalized coordinates. 

LDA seems to strike a bias-variance balance by learning the different covariances between features generally, without overfitting as quickly as QDA. 