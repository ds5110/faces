import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import column_or_1d

# Get data
infant = pd.read_csv('scratch/labels_decorated.csv')
adult = pd.read_csv('scratch/300w_valid_decorated.csv')

# Make infant and adult data same size
# infant_use = infant.loc[:, ['boxratio', 'boxsize/interoc']]
# adult_use = adult.loc[:, ['boxratio', 'boxsize/interoc']]
infant_use = infant.iloc[:, 8:]
adult_use = adult.iloc[:410, 4:]

X = pd.concat([infant_use, adult_use])
y = pd.DataFrame([1] * infant_use.shape[0] + [0] * adult_use.shape[0])
y = column_or_1d(y, warn=True)

# Train
pca = PCA(whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
n_components = range(5, 30, 5)

model = make_pipeline(pca, svc)

param_grid = {'svc__C': [1, 5, 10, 50, 100],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005],
              'pca__n_components': n_components}

grid = GridSearchCV(model, param_grid, return_train_score=True, n_jobs=5, refit=True)

grid.fit(X, y)

print(grid.best_params_)
print(f"best score: {grid.best_score_:.3f}")
