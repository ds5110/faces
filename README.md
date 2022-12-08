# DS5110 faces

This repo documents the development efforts in an exploratory data analysis and classical machine learning team project, as part of the classwork for DS5110, Fall 2022 at Northeastern University, Roux Institute.

## Data

This project leverages the adult and infant facial landmark data from this repo: [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking).

To download the data and generate the derived columns run the following command:
```
make merge_meta
```
To generate the euclidean pair-wise distance data run the following command:
```
make euclidian_data
```
## Reproducibility
To reproduce the results shown in this repository, first download the data and generate the additional features using the following two commands:
```
make merge_meta
make euclidian_data
```
To reproduce preprocessing steps: (see [preprocessing.md](https://github.com/ds5110/faces/blob/main/preprocessing.md) for details)
```
make prelim_plots
make roll_yaw
make angles_logreg
make angle_outliers
make compare_normalized
```

To reproduce resampling steps: (see [resampling.md](https://github.com/ds5110/faces/blob/main/sampling.md) for details)
```
make logreg_eda
make resample_test
```

To reproduce SVC model: (see [svc.md](https://github.com/ds5110/faces/blob/main/svc.md) for details)
```
make svc_geometric
make svc_landmarks
```

To reproduce logistic regression model: (see [logreg.md](https://github.com/ds5110/faces/blob/main/logreg.md) for details)
```
make logreg_eda
make make logreg_test
make logreg_4_pred

```

To reproduce Naive Bayes, LDA, QDA, & KNN models:
```
make logreg_eda
make resample_test
```

## Dependencies

There is a [requirements.txt](requirements.txt) file at the root of the repo, as a convenience for installing dependencies. It is strongly recommended to create a new virtual environment for this project (see [pip/venv documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for details). You can install these locally with pip:
  ```
  pip install -r requirements.txt
  ```
  *Note: This file is not a minimally sufficient list of dependencies; it most certainly includes extraneous packages that happen to be in our initial development environment.*

## Overview
* [Preprocessing](preprocessing.md)
* [Sampling](sampling.md)
* [Feature Selection](feature_selection.md)
* Training Models
  * [PCA > SVC](svc.md)
  * [Logistic Regression](logreg.md)
  * [LDA/QDA](lda.md)
