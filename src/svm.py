import numpy as np
import util.preprocessing as preprocessing
import util.analysis as analysis

from sklearn import svm
from sklearn import metrics
from sklearn import cross_validation


def run_svm(): 
	_, X, y = preprocessing.create_data_matrices('./data')

	#discretize y into buckets 
	y_disc, bins_pv, bins_pf = preprocessing.discretize_labels(y, return_bins=True)

	#split into train and test
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y_disc, test_size=0.3)

	##split y into pv and pf 
	y_train_pv, y_train_pf = y_train[:,0], y_train[:,1]
	y_test_pv, y_test_pf = y_test[:,0], y_test[:,1] 

	#fit svm model and predict

	case_predict("pv", "linear", "ovr", (X_train, y_train_pv, X_test, y_test_pv))
	case_predict("pf", "linear", "ovr", (X_train, y_train_pf, X_test, y_test_pf))


def case_predict(vector_type, kernel_s, dfs, datasets): 
	X_train, y_train, X_test, y_test = datasets[0], datasets[1], datasets[2], datasets[3]
	svm_model = svm.SVC(C=0.01, kernel=kernel_s, decision_function_shape=dfs)
	svm_model.fit(X_train, y_train)
	y_predict_train = svm_model.predict(X_train)
	y_predict = svm_model.predict(X_test)

	print "running analysis on results of " +  vector_type + " cases prediction..."
	analysis.run_analyses(y_predict_train, y_train, y_predict, y_test)

	print y_train.shape 
	print y_test.shape


if __name__ == '__main__':
	run_svm()


