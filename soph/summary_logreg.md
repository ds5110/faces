# DS5110 faces: Logistic Regression Exploration Summary - Sophia
The purpose of this file is to explain the main takeaways from the logreg model exploration.

For much more information about EDA, sampling methods attempted, hyperparameter tuning, etc please see `notes_logreg` and `notes_logreg_dist`.

## Logistic Regression: Take 1
The downsampled scores were consistanly better (even if it reduced the accuracy rating by 0.1 or so, the recall scores greatly improved), so we can conclude that downsampling is a good stragegy for this model. 

In the spirit of simple models being the "best", I would say that the one feature model is pretty good. I don't see much use for the two feature model (or the model that includes the "angular offsets"-see notes for more info) becuase it is very similar score-wise to the one feature model. 

If you really wanted those baby recall scores to be even higher, then the 13 feature model with the normalized & rotated coord data appears to be better than the just normalized coords given the similarity in scores and great reduction of features. This provides good evidence that the preprocessing adjustments done to the data are worthwile.

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
