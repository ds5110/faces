# Feature Selection

## Recursive Feature Selection
We used recursive feature selection to eliminate unnecessary features from data in order to improve model accuracy. Recusrive feature selection works by considsering a model with all features, and then removing certain features until a desired number remains.  

For example, logistic regression models need little to no multicollinearity in order to preform well. Using Recursive/Backward Feature Selection along with (cross validation to find the optimal number of features) allows us to remove multicollinearity and create a more accurate model. 

To see how our logistic regression model preformed with backward feature selection, refer to `logreg.md` for a summary and `notes\sc_notes_logreg.md` for a detailed comparison. We consistantly observed higher recall and accuracy scores using backward feature selection. 

## Forward Feature Selection
Part of our exploration involved calculating all of the euclidian-pairwise distances (see `preprocessing.md` for details). Since we have 67 (x,y) points, this clearly creates a lot of additional features (about x2 as many features than datapoints). Recursive feature seleciton wont work here for this reason, so we turned to another method: Forward Feature Selection. Forward feature selection starts with no features, and adds features to the model one by one until a new variable does not add to the improvement of the model.

For our logistic regression model, we decided to manually try 1, 2, 3, 4, 5, 6, and 7 features (as it starts taking a long time with 10+ features to consider). We got a steady improvement until around 4 features. Please see `sc_notes_dist_logreg.md` for details. 