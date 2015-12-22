import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics 


def output_error(y_predict, y_true):
        """
        Outputs several performance metrics of a given model, including precision,
        recall, f1score, and error.

        Args:
                y_predict: an array of the predicted labels of the examples
                y_true: an array of the true labels of the examples

        Returns
                (precision, recall, fscore, _), error
        """
        return metrics.precision_recall_fscore_support(y_true, y_predict), np.sum(y_predict != y_true) / float(y_predict.shape[0])


def run_analyses(y_predict_train, y_train, y_predict, y_test):
        """
        Runs analyses, including finding error, precision, recall, f1score,
        on the results of a particular model. Prints out the numeric metrics 
        and plots the graphical ones.

        Args:
                y_predict_train:
                        the predicted labels on the training examples
                y_train:
                        true labels on training examples
                y_predict:
                        predicted labels on testing examples
                y_test:
                        true labels on testing examples
                class_names:
                        dictionary that contains the class name that corresponds
                        with the class index

        Returns:
                None
        """
        # calculate metrics
        _, training_error = output_error(y_predict_train, y_train)
        (precision, recall, f1, _), testing_error = output_error(y_predict, y_test)
   
        # print out metrics
        print 'Average Precision:', np.average(precision)
        print 'Average Recall:', np.average(recall)
        print 'Average F1:', np.average(f1)
        print 'Training Error:', training_error
        print 'Testing Error:', testing_error


def run_regression_analyses(vector_type, X_test, labels, var_score, feature_list):
	y_predict_train, y_train, y_predict, y_test = labels[0], labels[1], labels[2], labels[3]
	test_residuals = y_predict - y_test


	for i in xrange(X_test.shape[1]): 
		plot_residuals(vector_type, X_test[:,i], test_residuals, feature_list[i])

	trainingMSE = np.mean((y_predict_train - y_train) ** 2)
	testingMSE = np.mean(test_residuals ** 2)
	print "training MSE:", trainingMSE
	print "testing MSE:", testingMSE
	print "variance score %2f" % var_score 


def plot_residuals(vector_type, x, residuals, feature):
	print "plotting residuals vs " + feature + " for " + vector_type + " cases..." 
	plt.scatter(x, residuals)
	plt.xlabel(feature)
	plt.ylabel('residuals')
	plt.title('residuals vs. ' + feature + ' for '+ vector_type + ' cases')
	plt.show()























