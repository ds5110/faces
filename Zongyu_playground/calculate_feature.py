import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import column_or_1d

# Get data
infant = pd.read_csv('Zongyu_playground/infant')
adult = pd.read_csv('Zongyu_playground/adult')
joint = pd.read_csv('Zongyu_playground/joint')

# Make infant and adult data same size
infant_use = infant.iloc[:, 9:]
adult_use = adult.iloc[:410, 5:]
adult_use.columns = infant_use.columns

X = pd.concat([infant_use, adult_use])
y = pd.DataFrame([1] * 410 + [0] * 410)
y = column_or_1d(y, warn=True)

# Train
pca = PCA(n_components=50, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')

model = make_pipeline(pca, svc)

param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}

grid = GridSearchCV(model, param_grid)

grid.fit(X, y)

print(grid.best_params_)
print(f"best score: {grid.best_score_:.3f}")
