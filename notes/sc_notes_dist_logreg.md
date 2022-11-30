# DS5110 faces: Notes on Logistic Regression Exploration (euclidean distance) - Sophia
The purpose of this file is to contain all the notes and testing related to logreg model exploration (take 2).

## EDA
We had the idea of trying to identify more "distances" between facial features that could help classify adults vs infants. We know from the paper and our original logreg exploration (take 1) that the ratio between the width and height of the head (boxratio) and distance between the eyes are good for this. Could there be more?

The EDA for this idea involved selecting one image and plotting the matrix of the euclidean distances. After generating the new data for all the points, I also attempted a correlation matrix of all the distances. I wanted to see if removing highly correlated points would make feature selection go faster, but I couldn't figure out the right way to do this that would still preserve the accuracy of the model. I ended up trying a new feature selection method (forward feature selection rather than backwards) and just dealing with the fact that it takes a long time to run the more features you consider. 

<img src="figs/soph_logreg/mat1.png" width=600>

<img src="figs/soph_logreg/mat2.png" width=600>

## Data used
Infant and adult raw coordinate data can be found from this repo: [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking).

## Preprocessing
For this model exploration I created additional metadata which computed the euclidean distances between the coordinate points (using normalized to the bounding box data as the coord input - see this document for information on how the normalized points were generated **link to different readme going over those preprocessing steps in more detail?**). Please see the `dist_metadata.py` for details on the calculations. To recreate the dataframe used, simply run `dist_metadata.py` and an updated csv will be created.

## Results
We approached this idea from 2 ways. We first wanted to see given **all** the information (euc distance from every point to every other point) what distances/features would the model choose, and how many would the model need until we got to a decent accuracy and recall score? We also were curious about what order it added those features in. 

**One feature**
Feature selected: ['dist_7_41'] (distance from the left chin to left eye)
* Accuracy score: 0.76 (no downsampling)
* Accuracy score: 0.83 (downsampling)
* Recall score adult: 0.92 (downsampling)
* Recall score baby: 0.74 (downsampling)

**Two feature**
Feature selected: ['dist_7_41', 'dist_22_25'] (new is right eyebrow width)
* Accuracy score: 0.85 (no downsampling)
* Accuracy score: 0.90 (downsampling)
* Recall score adult: 0.94 (downsampling)
* Recall score baby: 0.87 (downsampling)

**Three feature**
Feature selected: ['dist_7_41', 'dist_21_22', 'dist_22_25'] (new is distance between eyebrows)
* Accuracy score: 0.87 (no downsampling)
* Accuracy score: 0.93 (downsampling)
* Recall score adult: 0.96 (downsampling)
* Recall score baby: 0.89 (downsampling)

**Four feature**
Feature selected: ['dist_7_41', 'dist_21_22', 'dist_22_25', 'dist_33_65'] (new is distance from mouth to nose)
* Accuracy score: 0.93 (no downsampling)
* Accuracy score: 0.99 (downsampling)
* Recall score adult: 1.00 (downsampling)
* Recall score baby: 0.97 (downsampling)

<img src="figs/soph_logreg/algo_baby.png" width=600>

We then wanted to see given only some distances, would the classifier be able to generate a good score? We heuristically chose these landmarks based on what we thought would be valuable. The distances chosen are as follows:
Left(L) brow width(W)
Right(R) brow width(W)
Inner(I) brow distance(D)
Outer(O) brow D
Leye W
Reye W
I eye D
O eye D
Leye to Lbrow
Reye to Rbrow
Ear to ear
Chin to chin
Chin to nose
Mouth W 
Mouth height(H)
Mouth center to eye center
Nose center to Leye
Nose center to Reye
Lip to Lbrow
Lip to Rbrow
Lchin to REar
Lchin to mouth center
Lchin to nose
Lchin to Leye
Lchin to Reye
Chin to Mouth

Visual representation of what we picked:

<img src="figs/soph_logreg/h_baby.png" width=600>

**One feature**
Feature selected: ['dist_21_22'] (distance between inner brows)
* Accuracy score: 0.74 (no downsampling)
* Accuracy score: 0.76 (downsampling)
* Recall score adult: 0.82 (downsampling)
* Recall score baby: 0.70 (downsampling)

**Two feature**
Feature selected: ['dist_21_22', 'dist_48_54'] (mouth width)
* Accuracy score: 0.81 (no downsampling)
* Accuracy score: 0.86 (downsampling)
* Recall score adult: 0.94 (downsampling)
* Recall score baby: 0.78 (downsampling)

**Three feature**
Feature selected: ['dist_21_22', 'dist_48_54', 'dist_27_62'] (from between eyes to top of mouth)
* Accuracy score: 0.84 (no downsampling)
* Accuracy score: 0.93 (downsampling)
* Recall score adult: 0.96 (downsampling)
* Recall score baby: 0.89 (downsampling)

**Four feature**
Feature selected: ['dist_21_22', 'dist_17_26', 'dist_48_54', 'dist_27_62']
* Accuracy score: 0.85 (no downsampling)
* Accuracy score: 0.94 (downsampling)
* Recall score adult: .92 (downsampling)
* Recall score baby:  .95 (downsampling)

<img src="figs/soph_logreg/myh_baby.png" width=600>

## Reproduce Results
* To reproduce the results from these notes, run the `logreg_dist_ex.py` file (testing).
  * Note that this testing uses forward feature selection for the dimentional reduction method, so it does take time to run depending on how many features you ask it to select (still under 5 min on my machine).
* To reproduce the data used in these notes, run `?jesse file?` and `dist_metadata.py`
* To reproduce the eda used in these notes, run `sc_eda_euc_logreg.py`
* To reproduce the testing for the sampling used in these notes run `imbalanced_resampling.py`