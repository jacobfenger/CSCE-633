import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import csv
import sys
from svmutil import *
import time

# The feature numbers that need to be transformed in the data
t_features = [1, 6, 7, 13, 14, 15, 25, 28]

# Will transform a feature with 3 values to 3 features as 
# required in question 3.a
#
# This will replace feature at index i with 3 features in 
# the same index.
def transform_feature(row):

	for f in range(len(t_features)):
		idx = 1
		if t_features[f] != 1:
			idx = t_features[f]+(2*f)

		v = row[idx]
		row = np.delete(row, idx)

		if v == 1:
			row = np.insert(row, idx, [0, 0, 1])
		elif v == 0:
			row = np.insert(row, idx, [0, 1, 0])
		else:
			row = np.insert(row, idx, [1, 0, 0])

	return row

# Read the input file
def read_data(input_file):
	data = []

	with open(input_file, 'rb') as f:
		f = csv.reader(f)
		for row in f:
			s = np.asarray(map(float, row))
			s= transform_feature(s)
			data.append(s)
	
	return data


def linear_SVM(train, test, C):
	prob = svm_problem(train[1], train[0])
	param = svm_parameter()
	# param.kernel_type = 0 # 0 = linear kernel
	# param.C = 1
	# param.v = 3
	s = '-t 0 -c ' + str(C) + ' -v 3'
	m = svm_train(prob, s)

	#p_labs, p_acc, p_vals = svm_predict(test[1], test[0], m)
	#print p_acc[0]

	return m

def split_data(data):
	split = train_test_split(data, test_size = (len(data)/3), shuffle=False)
	train = split[0]
	test = split[1]

	train_y, train_x, test_y, test_x = [], [], [], []

	for row in train:
		train_y.append(row[-1])
		train_x.append(row[:-1].tolist())

	for row in test:
		test_y.append(row[-1])
		test_x.append(row[:-1].tolist())

	return train_x, train_y, test_x, test_y

def find_best_C(train, linear):

	C = [i for i in range(1, 150, 5)]
	acc = []
	training_times = []

	for c in C:
		if linear:
			t0 = time.clock()
			acc.append(linear_SVM(train, None, c))
			t1 = time.clock()

		training_times.append(t1-t0)

	plt.plot(C, acc, color='Grey')
	plt.xlim([0, max(C)+3])
	plt.title('3-Fold Cross Validation Accuracy for Different Values of Misclassification Cost C')
	plt.xlabel('C')
	plt.ylabel('Accuracy')
	plt.show()

	plt.plot(C, training_times, color='Blue')
	plt.xlim([0, max(C)+3])
	plt.title('Overall Cross Validation Training Time for Different Values of Misclassification Cost C')
	plt.xlabel('C')
	plt.ylabel('Cross Validation Time')
	plt.show()

def find_best_polynomial_kernel(train):
	prob = svm_problem(train[1], train[0])
	m = svm_train(prob, '-t 1 -d 7 -c 151 -g 0.125 -v 3')
	print 'Cross Validation Accuracy for Polynomial Kernel: %s' % m 

def find_best_RBF_kernel(train):
	prob = svm_problem(train[1], train[0])
	m = svm_train(prob, '-t 2 -c 151 -v 3 -g 0.125')
	print 'Cross Validation Accuracy for RBF Kernel: %s' % m 

# The best C for linear-SVM was 56
# Classification accuracy for test set was 92.08% for C = 56
def main():
	data = read_data('hw2_question3.csv')
	train_x, train_y, test_x, test_y = split_data(data)

	# Linear SVM
	#find_best_C((train_x, train_y), True)
	prob = svm_problem(train_y, train_x)
	m = svm_train(prob, '-t 0 -c 56')
	p_labs, p_acc, p_vals = svm_predict(test_y, test_x, m)
	print 'Testing Accuracy for Linear Kernel: %s ' % p_acc[0]

	# Kernel SVM
	find_best_polynomial_kernel((train_x, train_y))
	prob = svm_problem(train_y, train_x)
	m = svm_train(prob, '-t 1 -d 7 -c 151 -g 0.125')
	p_labs, p_acc, p_vals = svm_predict(test_y, test_x, m)
	print 'Testing Accuracy for Polynomial Kernel: %s ' % p_acc[0]

	# RBF Kernel 
	find_best_RBF_kernel((train_x, train_y))
	prob = svm_problem(train_y, train_x)
	m = svm_train(prob, '-t 2 -c 151 -g 0.125')
	p_labs, p_acc, p_vals = svm_predict(test_y, test_x, m)
	print 'Testing Accuracy for Polynomial Kernel: %s ' % p_acc[0]




if __name__ == '__main__':
	main()