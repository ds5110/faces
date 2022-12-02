# DS5110 faces: Notes on Logistic Regression Exploration - Sophia
The purpose of this file is to contain all the notes and testing related to logreg model exploration (take 1).

## EDA
I plotted the distribution of the "potential predictors" using histograms for both babys and adults. I was looking to see if any of the features seemed separable (thinking that would work well for logreg). I noted that the "boxratio" looked alright but the other ones did not look very separable.

<img src="soph_logreg/dist_p.png" width=600>

<img src="soph_logreg/dist_a.png" width=600>

## Data used
Infant and adult raw coordinate data can be found from this repo: [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking).

For this model exploration I used several "partitions" of the data. Information on how these partitions were created can be found [here](https://github.com/ds5110/faces/blob/SVC/derived.md).

* Partition 1: The "potential predictors" 
  * Aka `['boxratio', 'interoc','interoc_norm','boxsize','boxsize/interoc']`
* Partition 2: The "potential predictors" + "angular offsets" 
  * Aka `['boxratio', 'interoc','interoc_norm','boxsize','boxsize/interoc']` + `['yaw', 'yaw_abs','roll','roll_abs']`
* Partition 3: The "norm_cenrot-" columns
  * aka coords transformed to correct yaw and position & normalized to the new minimum bounding box
* Partition 4: The "norm_cenrot_sym_diff" columns
  * Aka the expected symmetric difference of yaw-corrected coordinates, normalized per min box

## Preprocessing
As outlined in the section above, there are several pre-processing steps done to create the "potential predictors" columns and transformed coordinates **(link to different readme going over those preprocessing steps in more detail?)**.

We also noticed that the dataset was somewhat unbalanced with more adult data points than baby data points. See `sample.md` for details on our resampling process.

## Results
**Partition 1: The "potential predictors"**
This partition produced pretty good results after feature selection and downsampling. The 2 features selected were `['boxratio', 'interoc']` (reinforcing paper findings).

Without feature selection:
* Accuracy score: 0.76
* Recall score adult: 0.84
* Recall score baby: 0.65

Confusion matrix:

<img src="soph_logreg/p1_nf_cmat.png" width=600>

With feature selection:
* Optimal features is 2
* Accuracy score: .89
* Recall score adult: 0.98
* Recall score baby: 0.76

CV to tune optimal features:

<img src="soph_logreg/p1_f_cv.png" width=600>

Confusion matrix:

<img src="soph_logreg/p1_f_cmat.png" width=600>

With feature selection & downsampling:
* Optimal features is 2 (unchanged)
* Accuracy score: 0.88
* Recall score adult: 0.90
* Recall score baby: 0.86

Confusion matrix:

<img src="soph_logreg/p1_fd_cmat.png" width=600>

**Partition 2: The "potential predictors"**
For this partition, I noted that there wasn't a huge increase from 1-2 features from Partition 1, so I decided to just use  `boxratio` to see what would happen. It produced similar results to above.

* Accuracy score: .89
* Recall score adult: 0.98
* Recall score baby: 0.77

Confusion matrix:

<img src="soph_logreg/p2_f_cmat.png" width=600>

With downsampling:
* Accuracy score: 0.88
* Recall score adult: 0.90
* Recall score baby: 0.85

Confusion matrix:

<img src="soph_logreg/p2_fd_cmat.png" width=600>

**Partition 3: The "potential predictors" + "angular offsets"**
For this partition, I wanted to see if the angular offsets would bring anything helpful to the table. I found that after feature selection the selected features were still `['boxratio', 'interoc']`, so I didn't include the rest of the results (as they are the same as partition 1 above).

Without feature selection:
* Accuracy score: 0.76
* Recall score adult: 0.84
* Recall score baby: 0.65

Confusion matrix:

<img src="soph_logreg/p3_nf_cmat.png" width=600>

With feature selection:
* Optimal features is 7
  * It is choosing 7, but after observing the CV plot I think the first 2 is good
* Accuracy score: .91
* Recall score adult: 0.99
* Recall score baby: 0.80

CV to tune optimal features:

<img src="soph_logreg/p3_f_cv.png" width=600>

**Partition 4: The "norm_cenrot" column**
This partition was pretty strong even before feature selection (is overfitting a possibility? It is a lot of features).

Without feature selection:
* Accuracy score: 0.96
* Recall score adult: 0.98
* Recall score baby: 0.95

Confusion matrix:

<img src="soph_logreg/p4_nf_cmat.png" width=600>

With feature selection:
* Optimal features is 81
  * It is choosing 81, but after observing the CV plot I think it really levels off around 66
* Accuracy score: .95
* Recall score adult: 0.98
* Recall score baby: 0.91

CV to tune optimal features:

<img src="soph_logreg/p4_f_cv.png" width=600>

Confusion matrix:

<img src="soph_logreg/p4_f_cmat.png" width=600>

With feature selection & downsampling:
* Optimal features is 66
* Accuracy score: 0.94
* Recall score adult: 0.92
* Recall score baby: 0.96

Confusion matrix:

<img src="soph_logreg/p4_fd_cmat.png" width=600>

**Partition 5: The "norm_cenrot_sym_diff" column**
This partition was also pretty strong even before feature selection (recall score on baby could be better) (is overfitting a possibility? It is a lot of features).

Without feature selection:
* Accuracy score: 0.95
* Recall score adult: 1.00
* Recall score baby: 0.87

Confusion matrix:

<img src="soph_logreg/p5_nf_cmat.png" width=600>

With feature selection:
* Optimal features is 13
* Accuracy score: .95
* Recall score adult: 0.98
* Recall score baby: 0.91

CV to tune optimal features:

<img src="soph_logreg/p5_f_cv.png" width=600>

Confusion matrix:

<img src="soph_logreg/p4_f_cmat.png" width=600>

With feature selection & downsampling:
* Optimal features is 13 (unchanged)
* Accuracy score: 0.94
* Recall score adult: 0.92
* Recall score baby: 0.96

Confusion matrix:

<img src="soph_logreg/p4_fd_cmat.png" width=600>

## Reproduce Results
To reproduce the results from these notes run:
```
make logreg_test
```
Note that this file has all 5 partitions and multiple combinations (feature selection, no feature selection, sampling, no sampling) so a lot gets output to the console and graphs generated. However it should run quickly and there are print statements that tell you what classificaition report you are looking at, etc.

To reproduce the data used in these notes, run `?jesse file?`

To reproduce the eda used in these notes run:
```
make logreg_eda
```
To reproduce the testing for the sampling used in these notes run:
```
make resample_test
```
