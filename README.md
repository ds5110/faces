# DS5110 faces

This repo records the development efforts in an exploratory data analysis team project, as part of the classwork for DS5110, Fall 2022 at Northeastern University, Roux Institute.

## Data

This project primarily leverages data from this repo: [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking).

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
  * [Decision Trees](tree.md)
  * [PCA > SVC](svc.md)
  * [Logistic Regression](logreg.md)
  * [LDA/QDA](lda.md)