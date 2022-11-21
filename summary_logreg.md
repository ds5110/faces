# DS5110 faces: Logistic Regression Exploration Summary - Sophia
The purpose of this file is to explain the main takeaways from the logreg model exploration.

For much more information about EDA, sampling methods attempted, hyper-parameter  tuning, etc please see `notes_logreg` and `notes_logreg_dist`.

## Logistic Regression: Take 1
The downsampled scores were consistently better (even if it reduced the accuracy rating by 0.1 or so, the recall scores greatly improved), so we can conclude that downsampling is a good strategy for this model. 

In the spirit of simple models being the "best", I would say that the one feature model is pretty good. I don't see much use for the two feature model (or the model that includes the "angular offsets"-see notes for more info) because it is very similar score-wise to the one feature model. 

If you really wanted those baby recall scores to be even higher, then the 13 feature model with the normalized & rotated coord data appears to be better than the just normalized coords given the similarity in scores and great reduction of features. This provides good evidence that the preprocessing adjustments done to the data are worthwhile.

One feature (`['boxratio']`), downsampling:
* Accuracy score: 0.88
* Recall score adult: 0.90
* Recall score baby: 0.85

Two feature (`['boxratio', 'interoc']`), downsampling:
* Accuracy score: 0.88
* Recall score adult: 0.90
* Recall score baby: 0.86

66 features (normalized coord data), downsampling:
* Accuracy score: 0.94
* Recall score adult: 0.92
* Recall score baby: 0.96

13 features (normalized & rotated coord data), downsampling:
* Accuracy score: 0.94
* Recall score adult: 0.92
* Recall score baby: 0.96

## Logistic Regression: Take 2
We found that the classifier produced a pretty good score with just one feature, the euclidian distance from the left chin to left eye.
* Accuracy score: 0.83 (downsampling)
* Recall score adult: 0.92 (downsampling)
* Recall score baby: 0.74 (downsampling)

Unlike in the previous exploration with the "potential predictors", adding more features did yeild improvements until 4 features.

Second (right eyebrow width)
* Accuracy score: 0.90 (downsampling)
* Recall score adult: 0.94 (downsampling)
* Recall score baby: 0.87 (downsampling)

Third (distance between eyebrows) 
* Accuracy score: 0.93 (downsampling)
* Recall score adult: 0.96 (downsampling)
* Recall score baby: 0.89 (downsampling)

Fourth (distance from mouth to nose)
* Accuracy score: 0.94 (downsampling)
* Recall score adult: .92 (downsampling)
* Recall score baby:  .95 (downsampling)

<img src="figs/soph_logreg/algo_baby.png" width=600>

## Logistic Regression: Bringing it together
In the first iteration of this exploration, we found that `boxratio` was a good predictor, but other single-features didn't add much to the accuracy of the model. In the second iteration, we found other distances that could be useful in classifying baby vs adult faces. Adding these features together (specifically `boxratio` with euclidean distance `'dist_7_41','dist_21_22', 'dist_22_25', 'dist_33_65'`) yeilded a model with:
* Accuracy score: 0.97
* Recall score adult: 0.97
* Recall score baby: 0.96

Interestingly, the feature selection considered all of them important because each additional distance added accuracy to the model. 

<img src="figs/soph_logreg/boxplus_f_cv.png" width=600>

I think this is a good model because the accuracy rate and recall scores are quite good and it doesn't require a ton of features (like the 13+ feature models from take 1).

<img src="figs/soph_logreg/boxplus_f_cmat.png" width=600>

<img src="figs/soph_logreg/boxplus_fd_cmat.png" width=600>
