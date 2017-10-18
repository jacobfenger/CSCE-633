################################################
# Implementation for problem 1.c) 
# Jacob Fenger
# 10/6/2017
################################################

import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read the input file
def read_data(input_file):
	data = []

	with open(input_file, 'rb') as f:
		f = csv.reader(f)
		for row in f:
			s = map(int, row)

			data.append(s)
	
	return np.asarray(data)

# Returns the proportion
def count_outcomes(data):

	benign = 0
	malignant = 0

	for s in data:
		if s[9] == 2: 
			benign += 1
		else:
			malignant += 1

	# print("Benign: ", benign, "Malignant:", malignant)
	# print("Proportion: ", float(benign)/malignant)

	return float(benign)/malignant


# Splits the data into equally proportioned training and testing sets
def split_data(data):

	count_outcomes(data)

	split = train_test_split(data, test_size = (len(data)/3), shuffle=False)
	train = split[0]
	test = split[1]

	# Not efficient, but it works I suppose 
	while abs(count_outcomes(train) - count_outcomes(test)) > 1:
		split = train_test_split(data, test_size = (len(data)/3), random_state=1, shuffle=True)
		train = split[0]
		test = split[1]

	count_outcomes(train)
	count_outcomes(test)

	return train, test

def main():
	data = read_data('hw2_question1.csv') 
	train, test = split_data(data)


if __name__ == '__main__':
	main()