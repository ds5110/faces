## Derived features

We added some additional metadata to the (original dataframe)[https://coe.northeastern.edu/Research/AClab/InfAnFace/labels.csv].

### Derived Landmark Coordinates
Where `dim` is 'x' or 'y' and `index` is an integer [0-67]

* `norm-{dim}{index}`
  * normalized to the minimum bounding box of landmarks
  * landmark 33 at image center
* `cenrot-{dim}{index}`
  * transformed to correct yaw and position
  * landmark 33 at image center
* `norm_cenrot-{dim}{index}`
  * transformed to correct yaw and position
  * normalized to the new minimum bounding box
  * landmark 33 at image center

### Differences of expected symmetric landmarks
Where `dim` is 'x' or 'y' and `index` is an integer [0-67] (the left-most point of the pair, viewer's perspective)
* `sym_diff-{dim}{index}`: difference of raw coordinates (probably not very useful)
* `cenrot_sym_diff-{dim}{index}`: difference of yaw-corrected coordinates
* `norm_cenrot_sym_diff-{dim}{index}`: difference of yaw-corrected coordinates, normalized per min box

### Angular offsets
* `yaw`: estimated rotation about the z-axis
* `yaw_abs`: magnitude of `yaw`
* `roll`: estimated rotation about the y-axis
* `roll_abs`: magintude of `roll`

### Predictors for distinguishing infants
* `boxratio`: width of min box divided by height of min box
* `interoc`: distance between outer canthi
* `interoc_norm`: `interoc` normalized per width/height of min box