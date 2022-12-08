merge_meta:
	python src/merge_meta.py

prelim_plots:
	python src/prelim_plots.py

angles_logreg:
	python src/angle_labels_logreg.py

roll_yaw:
	python src/plot_roll_yaw.py

angle_outliers:
	python src/big_yaw.py

logreg_4_pred:
	python src/logreg_4_pred.py

svc_scatter:
	python src/eg_svc.py

#creates merged_landmarks_dist.csv needed for generating pairwise euclidian distance metadata
euclidian_data:
	python src/util/sc_euc_metadata.py

#testing logistic regression function with different groups of features
logreg_test:
	python src/sc_logreg_ex.py

#testing logistic regression function with pairwise euclidian distance metadata
logreg_euc_test:
	python src/sc_logreg_euc_ex.py

#recreates figures used in logreg EDA
logreg_eda:
	python src/sc_eda_euc_logreg.py
	python src/sc_eda_logreg.py

#testing resample functions
resample_test:
	python src/sc_resample_test.py

# Result of SVC on landmarks
svc_landmarks:
	python src/zw_svc_landmarks.py

# Result of SVC on geometric values
svc_geometric:
	python src/zw_svc_geometric_value.py