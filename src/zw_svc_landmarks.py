import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Get data
df = pd.read_csv('data/merged_landmarks.csv')
X_original = df.loc[:, 'norm_cenrot-x0':'norm_cenrot-y67']
y = df.loc[:, 'baby']
X_train, X_test, y_train, y_test = train_test_split(X_original, y,
													test_size=0.33,
													random_state=42)

# Train on original
pca_original = PCA(whiten=True, random_state=42)
svc_original = SVC(class_weight='balanced')
model_original = make_pipeline(pca_original, svc_original)

param_grid_original = {'svc__C': [50, 100, 500, 1000],
					   'svc__gamma': [0.0001, 0.0005, 0.001, 0.005],
					   'pca__n_components': range(40, 70, 5),
					   'svc__kernel': ['rbf','linear', 'poly']}

grid_original = GridSearchCV(model_original, param_grid_original, return_train_score=True, n_jobs=5)
grid_original.fit(X_train, y_train)

print(grid_original.best_params_)
print(f"best CV score on train set: {grid_original.best_score_:.3f}\n")

# Predict
model_best = grid_original.best_estimator_
y_predict = model_best.predict(X_test)

# Get classification report
print('Classification report for best model:\n')
print(classification_report(y_test, y_predict, target_names=['adult', 'infant']))

# Plot confusion matrix
mat = confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='OrRd',
			xticklabels=['adult', 'infant'],
			yticklabels=['adult', 'infant'])
plt.title('SVC on landmarks')
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.savefig('figs/SVC_landmarks')
plt.show()

# Plot validation curve
pca_vali = PCA(whiten=True)
best_params_ = grid_original.best_params_
svc_vali = SVC(kernel=best_params_['svc__kernel'],
			   C=best_params_['svc__C'],
			   gamma=best_params_['svc__gamma'])
model_vali = make_pipeline(pca_vali, svc_vali)
param_range = range(2, 35)

train_scores, test_scores = validation_curve(
    model_vali,
    X_train,
    y_train,
    param_name="pca__n_components",
    param_range=param_range,
    scoring="f1",
    n_jobs=2,
)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve on landmarks")
plt.xlabel(r"$n\_components$")
plt.ylabel("F1_Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.plot(
    param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
)
plt.fill_between(
    param_range,
    train_scores_mean - train_scores_std,
    train_scores_mean + train_scores_std,
    alpha=0.2,
    color="darkorange",
    lw=lw,
)
plt.plot(
    param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
)
plt.fill_between(
    param_range,
    test_scores_mean - test_scores_std,
    test_scores_mean + test_scores_std,
    alpha=0.2,
    color="navy",
    lw=lw,
)
plt.legend(loc="best")
plt.savefig('figs/SVC_vali_landmarks')
plt.show()

# Print classification report for n_components = 15
pca_15 = PCA(whiten=True, random_state=42, n_components=15)
svc_15 = SVC(kernel=best_params_['svc__kernel'],
			   C=best_params_['svc__C'],
			   gamma=best_params_['svc__gamma'])
model_15 = make_pipeline(pca_15, svc_15)
model_15.fit(X_train, y_train)
y_predict_15 = model_15.predict(X_test)

print('Classification report for n_components = 15:\n')
print(classification_report(y_test, y_predict_15, target_names=['adult', 'infant']))


# Print classification report for n_components = 5
pca_5 = PCA(whiten=True, random_state=42, n_components=5)
svc_5 = SVC(kernel=best_params_['svc__kernel'],
			   C=best_params_['svc__C'],
			   gamma=best_params_['svc__gamma'])
model_5 = make_pipeline(pca_5, svc_5)
model_5.fit(X_train, y_train)
y_predict_5 = model_5.predict(X_test)

print('Classification report for n_components = 5:\n')
print(classification_report(y_test, y_predict_5, target_names=['adult', 'infant']))

# Print classification report for n_components = 2
pca_2 = PCA(whiten=True, random_state=42, n_components=2)
svc_2 = SVC(kernel=best_params_['svc__kernel'],
			   C=best_params_['svc__C'],
			   gamma=best_params_['svc__gamma'])
model_2 = make_pipeline(pca_2, svc_2)
model_2.fit(X_train, y_train)
y_predict_2 = model_2.predict(X_test)

print('Classification report for n_components = 2:\n')
print(classification_report(y_test, y_predict_2, target_names=['adult', 'infant']))

# Plot first two principal components
pca = PCA(whiten=True, n_components=2)
X_2d = pd.DataFrame(pca.fit_transform(X_original))
X_2d['PCA1'], X_2d['PCA2'], X_2d['baby'] = X_2d[0], X_2d[1], y
sns.lmplot(data=X_2d, x='PCA1', y='PCA2', hue='baby',legend=True, fit_reg=False)
plt.title('First 2 components true value')
plt.savefig('figs/PCA_of_landmarks_infant')
plt.show()

X_2d['pre'] = model_2.predict(X_original)
sns.lmplot(data=X_2d, x='PCA1', y='PCA2', hue='pre',legend=True, fit_reg=False)
plt.title('First 2 components predicted value')
plt.savefig('figs/PCA_of_landmarks_predict')
plt.show()