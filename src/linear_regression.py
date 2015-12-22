import numpy as np
import util.preprocessing as preprocessing
import util.analysis as analysis

from sklearn import linear_model 
from sklearn import metrics
from sklearn import cross_validation 

def run_linear_regression(): 
	feature_list, X, y = preprocessing.create_data_matrices('./data')

	#split into train and test
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)

	##split y into pv and pf 
	y_train_pv, y_train_pf = y_train[:,0], y_train[:,1]
	y_test_pv, y_test_pf = y_test[:,0], y_test[:,1] 

	case_predict("pv", (X_train, y_train_pv, X_test, y_test_pv), feature_list)	
	case_predict("pf", (X_train, y_train_pf, X_test, y_test_pf), feature_list)	


def case_predict(vector_type, datasets, feature_list): 
	X_train, y_train, X_test, y_test = datasets[0], datasets[1], datasets[2], datasets[3]
	l_regr = linear_model.LinearRegression() 

	#fit and prediction
	l_regr.fit(X_train, y_train)
	y_predict_train = l_regr.predict(X_train)
	y_predict = l_regr.predict(X_test)	

	print "running analysis on results of " +  vector_type + " cases prediction..."
	##calculate variance score 
	var_score = l_regr.score(X_test, y_test)
	analysis.run_regression_analyses(vector_type, X_test, (y_predict_train, y_train, y_predict, y_test), var_score, feature_list)



if __name__ == '__main__':
	run_linear_regression()
