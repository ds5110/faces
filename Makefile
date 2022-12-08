merge_meta:
	python src/jh_merge_meta.py

prelim_plots:
	python src/jh_prelim_plots.py

angles_logreg:
	python src/jh_angle_labels_logreg.py

compare_normalized:
	python src/jh_plot_norm.py

roll_yaw:
	python src/jh_plot_roll_yaw.py

angle_outliers:
	python src/jh_big_yaw.py

logreg_4_pred:
	python src/jh_logreg_4_pred.py

svc_scatter:
	python src/jh_eg_svc.py

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

=======
# Result of SVC with landmarks
svc_landmarks:
	python src/zw_svc_landmarks.py

# Result of SVC with geometric values
svc_geometric:
	python src/zw_svc_geometric_value.py
  
# recreates figures in bayes.md
bayes_figures:
  python src/cl_bayes.py

