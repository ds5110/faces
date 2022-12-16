import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Get data
df = pd.read_csv('data/merged_landmarks.csv')
X_geometric = df.loc[:, ['boxratio', 'interoc_norm']]
y = df.loc[:, 'baby']
X_train, X_test, y_train, y_test = train_test_split(X_geometric, y,
													test_size=0.33,
													random_state=42)

# Train on geometric values
svc_geometric = SVC(class_weight='balanced')

model_geometric = make_pipeline(svc_geometric)

param_grid_geometric = {'svc__C': [50, 100, 500, 1000],
						'svc__gamma': [0.05, 0.1, 0.5, 1, 5],
						'svc__kernel': ['rbf', 'linear', 'poly']}

grid_geometric = GridSearchCV(model_geometric, param_grid_geometric, return_train_score=True, n_jobs=5, refit=True)
grid_geometric.fit(X_train, y_train)

best_params_ = grid_geometric.best_params_
print(best_params_)
print(f"best train score: {grid_geometric.best_score_:.3f}\n")

# Predict
model_best = grid_geometric.best_estimator_
y_predict = model_best.predict(X_test)

# Get classification report
print('Classification report for model using geometric values:\n')
print(classification_report(y_test, y_predict, target_names=['adult', 'infant']))

# Plot confusion matrix
mat = confusion_matrix(y_test, y_predict)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, cmap='OrRd',
			xticklabels=['adult', 'infant'],
			yticklabels=['adult', 'infant'])
plt.title('SVC on geometric values')
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.savefig('figs/SVC_geometric_value')
plt.show()

# Compute support vectors
svc_trained = SVC(kernel=best_params_['svc__kernel'],
				  C=best_params_['svc__C'],
				  gamma=best_params_['svc__gamma'])
svc_trained.fit(X_train, y_train)
n_Support_vector = svc_trained.n_support_
print("Number of support vector: ", n_Support_vector)
Support_vector_index = svc_trained.support_

# Plot true values
X_geometric['baby'] = y
for i in X_geometric.index:
	X_geometric.loc[i, 'baby'] = 'adult' if X_geometric.loc[i, 'baby'] == 0 else 'infant'
sns.scatterplot(data=X_geometric, x='boxratio', y='interoc_norm',
				hue='baby',
				legend=True)
plt.title('All samples with true adult vs true infant')
plt.savefig('figs/Geometric_values')
plt.show()

# Plot predicted values
X_geometric['pre'] = svc_trained.predict(X_geometric.loc[:, ['boxratio', 'interoc_norm']])
for i in X_geometric.index:
	X_geometric.loc[i, 'pre'] = 'adult' if X_geometric.loc[i, 'pre'] == 0 else 'infant'
sns.scatterplot(data=X_geometric, x='boxratio', y='interoc_norm',
				hue='pre',
				legend=True)
plt.title('All samples with predicted adult vs predicted infant')
plt.savefig('figs/Geometric_values_pre_vs_true')
plt.show()

# Plot support vectors
support_vector = X_train.iloc[Support_vector_index][:]
support_vector['baby'] = y_train.iloc[Support_vector_index][:]
for i in support_vector.index:
	support_vector.loc[i, 'baby'] = 'adult' if support_vector.loc[i, 'baby'] == 0 else 'infant'
sns.scatterplot(data=support_vector, x='boxratio', y='interoc_norm',
				hue='baby',
				legend=True)
plt.title('Scatter plot of support vectors with geometric values')
plt.savefig('figs/Support_Vectors')
plt.show()

# Plot heatmap for geometric values
interoc_norm = np.linspace(0.1, 0.9, 81)
boxratio = np.linspace(0.7, 1.7, 101)
heatmap = [['.'] * len(boxratio) for _ in range(len(interoc_norm))]

for i in range(len(interoc_norm)):
	for j in range(len(boxratio)):
		heatmap[i][j] = model_best.predict(pd.DataFrame({'boxratio': [boxratio[j]],
														'interoc_norm': [interoc_norm[i]]}))[0]
heatmap = pd.DataFrame(heatmap)
heatmap.columns = boxratio
heatmap.index = interoc_norm
sns.heatmap(data=heatmap,
			cbar=False,
			xticklabels=10,
			yticklabels=10).invert_yaxis()

plt.xlabel('boxratio')
plt.ylabel('interoc_norm')
plt.title('Heatmap on meshgrid with geometric values')
plt.savefig('figs/SVC_meshgrid_of_geometric')
plt.show()
