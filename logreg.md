## Logistic Regression

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