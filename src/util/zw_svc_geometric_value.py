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
					   'svc__kernel': ['rbf','linear', 'poly']}

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


# Plot geometric values and support vectors
X_geometric['baby'] = y
sns.scatterplot(data=X_geometric, x='boxratio', y='interoc_norm',
				hue='baby',
				legend=True)
plt.savefig('figs/Geometric_values')
plt.show()

support_vector = X_train.iloc[Support_vector_index][:]
support_vector['baby'] = y_train.iloc[Support_vector_index][:]
sns.scatterplot(data=support_vector, x='boxratio', y='interoc_norm',
				hue='baby',
				legend=True)
plt.savefig('figs/Support_Vectors')
plt.show()

# Plot heatmap for geometric values
interoc_norm = np.linspace(0.1, 0.9, 100)
boxratio = np.linspace(0.7, 1.7, 100)
heatmap = [['.'] * len(boxratio) for _ in range(len(interoc_norm))]

for i in range(len(interoc_norm)):
	for j in range(len(boxratio)):
		heatmap[i][j] = model_best.predict([[boxratio[j], interoc_norm[i]]])[0]
heatmap = pd.DataFrame(heatmap)

sns.heatmap(data=heatmap,
			cbar-False)
plt.savefig('figs/SVC_meshgrid_of_geometric')
plt.show()