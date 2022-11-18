# DS5110-cv

## Data

This project uses data from this repo: [Infant-Facial-Landmark-Detection-and-Tracking](https://github.com/ostadabbas/Infant-Facial-Landmark-Detection-and-Tracking).

In order to minimize redundant downloads, this repo includes a [Makefile](Makefile) with a **data** target. Executing the **data** target will create a git-ignored sub-directory named "data", and download the required files into that location. The Python scripts that use the data expect the files are already downloaded and that the current working directory is the root of the repo.

There is also a convenience utility for loading these files, wrapped up in a utility module [util.local_cache](src/util/local_cache.py) that provides an interface `LocalCache` and default instance `cache` for accessing images and metadata, downloading them if they are not present locally.

## Local Execution

Local execution entails a few considerations that may be unintuitive:

* The current working directory is assumed to be the root of the repo. The intended location for cached images and metadata files is in the **data** directory. Of course, you can ignore this if you aren't using the [local_cache](src/util/local_cache) module (nor the similar [alt_cache](src/util/alt_cache) module). If you do want to use these, default instances are provided directly in the `util` package util [init](src/util/__init__.py), imported as `from util import cache` (for baby face metadata) and `from util import alt` (for adult face metadata).
* The **src** directory should be on the Python path, so that the "driver" scripts can locate the "library" modules in `util`. If you don't want to worry about this, you can work on scripts directly located under the **src** directory (Python path automatically includes the directory of the script being run).
* Command line execution requires running each script as a module and adding **src** to the path, e.g.:
  ```
  PYTHONPATH=./src python -m src.test.eg_logreg
  ```
* There is also a [requirements.txt](requirements.txt) file at the root of the repo, as a convenience for installing dependencies. It is strongly recommended to create a new virtual environment for this project (see [pip/venv documentation](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) for details). You can install these locally with pip:
  ```
  pip install -r requirements.txt
  ```
  *Note: This file is not a minimally sufficient list of dependencies; it most certainly includes extraneous packages that happen to be  environment.*

## Preprocessing

(For a full list of derived features, see [derived](derived.md).)

In our initial exploration we tried plotting landmarks on a frontal view example image:

<img src="figs/images/ads/14998.Still056.jpg_scatternum.png" width=600>

Then we tried drawing splines between subsets of points, to make detected features more clear:

<img src="figs/images/ads/14998.Still056.jpg_spline.png" width=600>

We also tried to standardize the orientation by centering and rotating about the z-axis (using expected symmetry between eye landmarks to calculate the angle of rotation):

<img src="figs/images/ads/14998.Still056.jpg_rotated.png" width=600>

To confirm this, we applied this rotation/centering on a few other images:

<img src="figs/images/ads/13555.Still049.jpg_scatter_rotated.png" width=600>
<img src="figs/images/ads/1072.Still005.jpg_splinelabel_rotated.png" width=600>

Depending on the input data and type of models we are training, this type of preprocessing may be useful. See [angular](angular.md) doc for more info on angular transformation analysis.

## Other EDA
We also wanted to investigate some basic stats about the data, such as how many images are in the sets, and how many different images have the targets listed. Using the script `combos.py`, we found that there are more images in the adds google and youtube sets, with less in the google2 and youtube2 sets (I think this makes sense as the 2 sets are for testing). We also found out that there are more images with just one of the targets, like just tilted or just expressive. Less images are marked with multiple, and no images are marked with 'turned', 'occluded', and 'expressive'.

<img src="figs/combos1.png" width=900>
<img src="figs/combos2.png" width=900>

We preformed a check using `plot_shape.py` to see if the images were all the same size (same number of rows and columns) in order to find out if we needed to scale the values for further EDA. We found that they are the same size, see the histogram and scatterplot below. 

<img src="figs/hist.png" width=900>
<img src="figs/scatter.png" width=900>

Because one idea for this project is creating a classifier of adult and infant faces, we also wanted to explore some of the adult face data. Using a similar script to `plot_gxgy.py`, we used `plot_adult.py` to plot the features.

<img src="figs/adult.png" width=900>
