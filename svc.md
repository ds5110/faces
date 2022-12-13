## SVC

Applied SVC to landmarks and to geometric values. 

`GridSearchCV` in both methods are applied to the train set, and classification reports are generated from the test set.

Landmarks are already normalized and geometric values are computed from the normalized landmarks.

Can reproduce all results by using `make svc_landmarks` and `make svc_geometric`, or simply run `src/zw_svc_landmarks.py` and `src/zw_svc_geometric_value.py`.

### Landmarks

There are 68 landmarks and two axis, x and y, for each landmark so there are 136 predictors in total.

Considering the number of predictors, we first applied `PCA` to reduce the number of dimension and then used `SVC` for classification. `GridSearchCV` is used to find the best hyperparameters for both `PCA` and `SVC`.

Bellow is the classification report on landmarks.

`n_components = 55`
- Accuracy: 0.99
- adult f1-score: 0.99
- infant f1-score: 0.99

The score of landmarks could be an upper bound for model performance since it contains all information in the landmarks. 

This performance is already high enough. Even if it might be further improved, the improvement will be tiny. Preprocessing like centering, rotating to upright position and scaling are needed here.

Bellow is the confusion matrix of classification.

<img src="figs/SVC_landmarks.png">

Since the above two graphs are for `n_components = 55` which is a quite big number, we also draw a validation curve on `n_components`. The `F1_score` is actually pretty good when `n_components = 15`

<img src="figs/SVC_vali_landmarks.png">

So we printed more classification reports with different number of principal component. Other hyperparameters are the same as in the model with `n_components = 55`.

`n_components = 15`
- Accuracy: 0.98
- adult f1-score: 0.99
- infant f1-score: 0.98

`n_components = 5`
- Accuracy: 0.84
- adult f1-score: 0.88
- infant f1-score: 0.76

`n_components = 2`
- Accuracy: 0.74
- adult f1-score: 0.82
- infant f1-score: 0.53

The score for `n_components = 15` is still pretty good, and `n_components = 5` is not that bad. But `n_components = 2` is simply guessing for the infant samples.

As the scatter plot of the first two principals shows bellow, the model with `n_components = 2` just predict the lower half part to be andult.

<img src="figs/PCA_of_landmarks_infant.png">
<img src="figs/PCA_of_landmarks_predict.png">

All above results are based on normalized landmarks. Bellow we add some results for using not normalized landmarks.

`n_components = 55`
- Accuracy: 0.98
- adult f1-score: 0.98
- infant f1-score: 0.97

`n_components = 15`
- Accuracy: 0.98
- adult f1-score: 0.98
- infant f1-score: 0.97

`n_components = 5`
- Accuracy: 0.78
- adult f1-score: 0.85
- infant f1-score: 0.61

`n_components = 2`
- Accuracy: 0.75
- adult f1-score: 0.83
- infant f1-score: 0.54

And the first two principal components using not normalized landmarks.

<img src="figs/PCA_of_landmarks_infant_not_normalized.png">
<img src="figs/PCA_of_landmarks_predict_not_normalized.png">

### Geometric Values

Now we use the two geometric values, `boxratio` and `interoclar_norm`, as predictors. Those two predictors are computed from the normalized landmarks, and they correspond to the `box width / box height` and `box size / interocular didtance` in `TABLE II` of Dr. Wan's paper. 

Bellow is the performance on geometric values. The classification report is generated on the test set, and the last lines shows the number of support vectors in the train set with 736 samples.

- Accuracy: 0.91
- adult f1-score: 0.93
- infant f1-score: 0.87
- Number of support vector: 80 (adult), 78 (infant)

The performance on the two geometric values is nice, and it's between the performance with `n_components = 5` and `n_components = 15` in the model using landmarks. This could serve as a lower bound since it requires only the ratio of box width to box height and box size to interocular distance.

Little preprocessing is needed given the bounding box and landmarks. It’s also more interpretable as we know the meaning of the predictors.

Bellow is the confusion matrix of classification.

<img src="figs/SVC_geometric_value.png">

Here is a scatter plot with `boxratio` and `interoc_norm` as axis. The points tend to be in two groups though there are some overlapping. 

<img src="figs/Geometric_values.png">

And if we draw the same scatter plot with predicted value. 

<img src="figs/Geometric_values_pre_vs_true.png">

Same plot on a meshgrid.

<img src="figs/SVC_meshgrid_of_geometric.png">


The support vectors in the train set. They tend to form a curved boundary between infant and adult samples. 

<img src="figs/Support_Vectors.png">

