# DS5110 faces: Notes on Logistic Regression Exploration (euclidean distance) - Sophia

## EDA
We had the idea of trying to identify more "distances" between facial features that could help classify adults vs infants. We know from the paper and our original logreg exploration that the ratio between the width and height of the head (boxratio) and distance between the eyes are good for this. Could there be more?

The EDA for this idea involved selecting one image and plotting the matrix of the euclidean distances. After generating the new data for all the points, I also attemped a correlation matrix of all the distances. I wanted to see if removing highly correlated points would make feature selection go faster, but I couldn't figure out the right way to do this that would still preserve the accuracy of the model. I ended up trying a new feature selection method (forward feature selection rather than backwards) and just dealing with the fact that it takes a long time to run the more features you consider. 

## Data used
For this model exploration I created additional metadata which computed the euclidean distances between the coordinate points (using Jesse's normalized to the bounding box data as the coord input).

## Results


## 