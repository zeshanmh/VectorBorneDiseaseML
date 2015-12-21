import numpy as np 
import pandas as pd
import math 
import os

NUM_REGIONS = 35
NUM_YEARS = 20

def create_data_matrices(basepath='../data'): 
	"""
	Loads the data into a dataframe and parses it into an input 
	and output matrix.

	Args:
		basepath: base path of data; default is '../data'

	Returns:
		X_full: input matrix (unflattened); dimension of input 
		matrix is (NUM_REGIONS, NUM_FEATURES, NUM_YEARS), where 
		NUM_REGIONS = 35, NUM_FEATURES = ?, NUM_YEARS = 20

		y_full: input label matrix (unflattened); dimension of
		input label matrix is (NUM_REGIONS, NUM_YEARS). Consider
		only last column of matrix

		region_dictionary: dictionary where key is index and value is
		name of state in India corresponding to the index

	"""

	#load data as dataframe 
	df = pd.read_csv(basepath + '/malaria_endemicity_data.csv')
	
	##store labels as y matrix (if using tensor, only consider last column of y_full)
	labeled_data = df.as_matrix()
	y_full = labeled_data[:,1:]
	region_dictionary = [labeled_data[i,0] for i in xrange(labeled_data.shape[0])]

	region_dictionary = {}
	for i in xrange(labeled_data.shape[0]): 
		region_dictionary[i] = labeled_data[i,0]

	return y_full, region_dictionary


def flatten_matrices(y_full): #need to add X
	"""
	Flattens y_full and X_full. We essentially treat the API at each 
	year for each region as an independent example.

	Args:
		X_full: input matrix (unflattened); dimension of input 
		matrix is (NUM_REGIONS, NUM_FEATURES, NUM_YEARS), where 
		NUM_REGIONS = 35, NUM_FEATURES = ?, NUM_YEARS = 20
		
		y_full: input label matrix (unflattened); dimension of
		input label matrix is (NUM_REGIONS, NUM_YEARS).

	Returns:
		X_flatten: flattened input matrix whose dimensions are 
		(NUM_REGIONS*NUM_YEARS, NUM_FEATURES)

		y_flatten: flattened input labels whose dimensions are 
		(NUM_REGIONS*NUM_YEARS,)

	"""

	return y_full.flatten()	


def get_region_value(idx):
	"""
	Computes the corresponding index value of the region 
	in the dictionary

	"""
	return idx / NUM_YEARS + 1


def test(): 
	y_full, region_dictionary = create_data_matrices()

	#flatten
	original = y_full[0,:]
	y_full = flatten_matrices(y_full)
	new_column = y_full[0:NUM_YEARS,]

	assert original.any() == new_column.any()
	assert y_full.shape == (NUM_REGIONS*NUM_YEARS,)


if __name__ == '__main__':
	test()