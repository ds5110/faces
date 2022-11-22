# Resampling
We noticed that the dataset was somewhat unbalanced with more adult data (689) points than baby data (410) points. Generally classification models preform better with balanced data, so we wanted to provide some options such as upsampling and downsampling. 

We also looked at specific metrics such as recall score, and plotted the confusion matrix before and after resampling to see if resampling allowed for less false negatives and false positives. The idea here is that we want to ensure that our model isn't "missing" too many instances of the minority class.

<img src="figs/soph_logreg/unbal.png" width=600>

`resample.py`

## Upsampling
Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.

## Downsampling

Logreg Examples:

**Two features**
Before downsampling:
* Accuracy score: .89
* Recall score adult: 0.98
* Recall score baby: 0.76

Confusion matrix:

<img src="figs/soph_logreg/p1_f_cmat.png" width=600>

After downsampling:
* Optimal features is 2 (unchanged)
* Accuracy score: 0.88
* Recall score adult: 0.90
* Recall score baby: 0.86

Confusion matrix:

<img src="figs/soph_logreg/p1_fd_cmat.png" width=600>