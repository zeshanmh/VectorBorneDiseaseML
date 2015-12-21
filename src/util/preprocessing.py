import numpy as np 
import pandas as pd
import math 
import os


NUM_YEARS = 20

def create_data_matrices(basepath='../data'): 
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
	return y_full.flatten()	


def get_region_value(idx):
	return idx / NUM_YEARS + 1


def test(): 
	y_full, region_dictionary = create_data_matrices()

	#flatten
	original = y_full[0,:]
	y_full = flatten_matrices(y_full)
	new_column = y_full[0:20,]

	assert original.any() == new_column.any()
	assert y_full.shape == (700,)


if __name__ == '__main__':
    test()