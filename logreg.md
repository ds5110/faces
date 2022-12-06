# DS5110 faces: Logistic Regression Summary

The purpose of this file is to explain the main takeaways from the logreg model exploration.

More information about EDA, sampling methods attempted, hyper-parameter tuning, and data partitions tested please see `notes/notes_logreg` and `notes/notes_logreg_dist`.

## Logistic Regression: Take 1

We tried using just four likely predictors "yaw_abs", "roll_abs", "boxratio", and "interoc_norm" to predict target "baby".

```
=====================================================
Classification Report
=====================================================
              precision    recall  f1-score   support

        baby       0.90      0.96      0.93       170
       adult       0.94      0.83      0.88       105

    accuracy                           0.91       275
   macro avg       0.92      0.90      0.91       275
weighted avg       0.91      0.91      0.91       275

	logreg score: 0.913
	coefficients:
		boxratio: 9.836838008155665
		yaw_abs: 3.2714741040479107
		interoc_norm: 1.4026847522362897
		roll_abs: 0.7687748782704812
	dummy score: 0.484
```

We plotted these results on a scatter of 'boxratio' vs 'yaw_abs' and a confusion matrix of the (small) validation set:

<img src="figs/scatter_boxratio_vs_yaw.png" width=600>
<img src="figs/logreg_4_pred_conf_mat.png" width=600>

This is a rough first test using small random validation set, but it indicates that 'boxratio' is likely useful. We imagine 'yaw_abs' is _circumstantially_ useful, since babies are generally photographed from more extreme angles, if only because they have a hard time supporting their relatively large heads.

## Logistic Regression: Take 2
One of our goals is to test out different forms of resampling to balance the data. We wanted to know if resampling would have a positive impact on model accuracy. We found that "downsampled" scores were consistently better (even if it reduced the accuracy rating by 0.1 or so, the recall scores greatly improved), so we can conclude that downsampling is a good strategy for this model. 

Building off of our findings in Take 1 above, the "one feature" model using just `['boxratio']` stood out to us as pretty good. For more information on how this model preformed with other groups of features, please see `notes/sc_notes_logreg`.

In many of these instances, we did recursive feature selection to select the best (and minimum) number of features. Espeically in the cases where we are using 66+ features, we found that using feature selection greatly improved both the accuracy and recall scores of our model. This is likely due to logistic regression preforming better with minimal colinearity. 

One feature (`['boxratio']`), downsampling:
* Accuracy score: 0.88
* Recall score adult: 0.90
* Recall score baby: 0.85

CV to tune optimal features:

<img src="figs/soph_logreg/p1_f_cv.png" width=600>

Confusion matrix:

<img src="figs/soph_logreg/p1_fd_cmat.png" width=600>

We beleive this model preformed well due to the preproccing adjsutments that went into the `boxratio` predictor (the data was normalized, see `preprocessing.md` for details), and the fact that this feature is seperable:

<img src="figs/soph_logreg/dist_p.png" width=600>

To reproduce the results from this section run:
```
make logreg_test
```
```
make logreg_eda
```
## Logistic Regression: Take 3
Inspired by our sucess with using `boxratio` (ratio between the width and height of the head) as a single predictor, we had the idea of trying to identify more "distances" between facial features that could help classify adults vs infants.

After generating the pairwise euclidian distances, it became really obvious that we would need feature selection given the ammount of features. In fact, if you tried to run this model with all these new features, you run the risk of overfitting the model. We thought that forward feature selection would be a good choice, as we can easily identify how and which additional distances will imporve the model. 

We also used downsampling here given the sucess of downsampling with the previous set of features.


We found that the classifier produced a pretty good score with just one feature:
**One feature**
Feature selected: ['dist_7_48'] (distance from the left chin to left mouth corner)
* Accuracy score: 0.74 (no downsampling)
* Accuracy score: 0.82 (downsampling)
* Recall score adult: 0.86 (downsampling)
* Recall score baby: 0.78 (downsampling)

We saw that adding more features did yeild improvements until 4 around features.
**Two feature**
Feature selected: ['dist_7_48', 'dist_33_42'] (new is nose to right eye corner)
* Accuracy score: 0.83 (no downsampling)
* Accuracy score: 0.91 (downsampling)
* Recall score adult: 0.93 (downsampling)
* Recall score baby: 0.89 (downsampling)

**Three feature**
Feature selected: ['dist_7_48', 'dist_21_22', 'dist_33_42'] (new is distance between eyebrows)
* Accuracy score: 0.91 (no downsampling)
* Accuracy score: 0.94 (downsampling)
* Recall score adult: 0.98 (downsampling)
* Recall score baby: 0.89 (downsampling)

**Four feature**
Feature selected: ['dist_7_48','dist_18_25','dist_21_22', 'dist_33_42'] (new is distance from mid eyebrow to mid eyebrow)
* Accuracy score: 0.94 (no downsampling)
* Accuracy score: 0.98 (downsampling)
* Recall score adult: 0.98 (downsampling)
* Recall score baby: 0.98 (downsampling)

**Five feature**
Feature selected: ['dist_7_9', 'dist_7_48', 'dist_18_25', 'dist_21_22', 'dist_33_42']
* Accuracy score: 0.95 (no downsampling)
* Accuracy score: 0.98 (downsampling)
* Recall score adult: .98 (downsampling)
* Recall score baby:  .97 (downsampling)

**Six feature**
Feature selected: ['dist_5_7', 'dist_7_9', 'dist_7_48', 'dist_18_25', 'dist_21_22', 'dist_33_42']
* Accuracy score: 0.97 (no downsampling)
* Accuracy score: 0.98 (downsampling)
* Recall score adult: .98 (downsampling)
* Recall score baby:  .98 (downsampling)

<img src="figs/soph_logreg/6_baby.png" width=600>

To reproduce the results from this section run:
```
make logreg_euc_test
```
## Logistic Regression: Bringing it together
In the initial iterations of this exploration, we found that `boxratio` was a good single predictor. In the second iteration, we found other distances that could be useful in classifying baby vs adult faces. Adding these features together (specifically `boxratio` with euclidean distance `'dist_5_7', 'dist_7_9', 'dist_7_48', 'dist_18_25', 'dist_21_22', 'dist_33_42'`) yeilded a model with:
* Accuracy score: 0.96
* Recall score adult: 0.99
* Recall score baby: 0.93

Interestingly, the feature selection considered all of them important because each additional distance added accuracy to the model. 

<img src="figs/soph_logreg/p3_cv.png" width=600>

I think this is a good model because the accuracy rate and recall scores are quite good and it doesn't require a ton of features.

<img src="figs/soph_logreg/p3_f_cmat.png" width=600>

<img src="figs/soph_logreg/p3_fd_cmat.png" width=600>

Similar to above, we beleive this model preformed well due to the preproccing adjsutments that went into the `boxratio` and euclidian distances (the data was normalized, see `preprocessing.md` for details), and the fact that these features are somewhat seperable:

<img src="figs/soph_logreg/dist_h.png" width=600>

To reproduce the results from this section run:
```
make logreg_test
```
```
make logreg_eda
```
