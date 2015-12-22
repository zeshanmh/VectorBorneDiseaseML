import numpy as np 
import pandas as pd
import math 
import os

NUM_REGIONS = 35
NUM_BUCKETS = 50
NUM_YEARS = 20

def create_data_matrices(basepath='../data'): 
	"""
	Loads the data into a dataframe and parses it into an input 
	and output matrix.

	Args:
		basepath: base path of data; default is '../data'

	Returns:
		feature_list: list of features, including rainfall 

		X: input matrix; dimension of input matrix is (NUM_EXAMPLES, 
			NUM_FEATURES) where NUM_EXAMPLES = 84 and NUM_FEATURES = 13

		y: input label matrix; number of malaria cases caused by 
		pf and pv mosquito. Dimension of matrix is (NUM_EXAMPLES, 2)
	"""

	#load data as dataframe 
	df = pd.read_csv(basepath + '/district_malaria_data.csv')
	feature_list = list(df.axes[1])

	labeled_data = df.as_matrix()
	X = labeled_data[:,:-2]
	y = labeled_data[:,-2:]
	return feature_list, X, y


def create_train_test_split(X, y, test_size=0.3, shuffle=False, debug=False): 
	"""
	Splits data into a training and a testing set.

	Args:
		X: data tensor created by create_data_tensor
		y: labels array created by create_data_tensor
		test_size: size of test set. default: 30%
		shuffle: boolean defining if the sets should be shuffled
			before being split

	Returns:
		train_X:
			data tensor containing training samples
		train_y:
			array containing class labels for training samples
		test_X:
			data tensor containing testing samples
		test_y:
			array containing class labels for testing samples
	"""
	class_labels = np.unique(y)
	train_samples_idx = []
	test_samples_idx = []

	# Shuffle the data
	if shuffle:
		if debug: print 'Shuffling data...',
		shuffled_idx = np.arange(y.shape[0])
		np.random.shuffle(shuffled_idx)
		X = X[shuffled_idx, :]
		y = y[shuffled_idx, :]

		if debug: print 'done'
	
	# Collect all the indices for the training and testing set
	if debug: print 'Extracting sample indices...',
	for label in class_labels:
		idx = np.where(y[:,0]==label)[0]
		train_samples = int(len(idx) * (1-test_size))
		test_samples = len(idx) - train_samples

		train_samples_idx.append(idx[:train_samples])
		test_samples_idx.append(idx[train_samples:])
	if debug: print 'done'

	# Concatenate all indices arrays into a single array of indices
	if debug: print 'Collecting indices...',
	train_samples_idx = np.concatenate(tuple(train_samples_idx))
	test_samples_idx = np.concatenate(tuple(test_samples_idx))
	if debug: print 'done'

	if shuffle:
		if debug: print 'Shuffling collecting indices...',
		np.random.shuffle(train_samples_idx)
		np.random.shuffle(test_samples_idx)
		if debug: print 'done'

	# Index and return the correct sets
	if debug: print 'Returning sets...',
	return X[train_samples_idx, :], y[train_samples_idx, :], X[test_samples_idx, :], y[test_samples_idx, :]


def discretize_labels(y, return_bins=False): 
	max_val1, min_val1 = y[:,0].max(), y[:,0].min() 
	bins1 = np.linspace(min_val1, max_val1, NUM_BUCKETS) ##bins represent upper bounds 
	class_labels1 = np.digitize(y[:,0], bins1) ##y_pv
	
	max_val2, min_val2 = y[:,1].max(), y[:,1].min()
	bins2 = np.linspace(min_val2, max_val2, NUM_BUCKETS)
	class_labels2 = np.digitize(y[:,1], bins2) ##y_pf

	print "y:", y 
	print "bins pv:", bins1
	print "discretized labels pv:", class_labels1

	print "bins pf:", bins2
	print "discretized labels pf:", class_labels2

	y_discretized = np.array([class_labels1, class_labels2]).T
	if return_bins: 
		return y_discretized, bins1, bins2

	return y_discretized

	# print np.concatenate(np.array([class_labels1]), np.array([class_labels2]))

def test(): 
	print 'Running tests for create_train_test_split'
	X = np.random.rand(10,40)
	y = np.array([[0,0],[1,1],[2,2],[3,3],[1,1],[4,4],[4,4],[2,2],[3,3],[4,4]], dtype=np.uint)
	tests = []
	for i in xrange(0,5):
		idx = np.where(y[:,0] == i)[0]
		test = np.sort(X[idx, :], axis=0)
		tests.append(test)

	X_train, y_train, X_test, y_test = create_train_test_split(X, y, test_size=0.5, shuffle=True)

	for i in xrange(0,5):
		test2 = np.sort(np.concatenate((X_train[np.where(y_train[:,0] == i)[0], :], X_test[np.where(y_test[:,0] == i)[0], :])), axis=0)
		assert(np.sum(tests[i] == test2) == np.size(tests[i]))


def test2(): 
	feature_list, X, y = create_data_matrices()
	y_discretize, bins = discretize_labels(y, True)

	


if __name__ == '__main__':
	test2()