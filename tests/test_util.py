import numpy as np
import matplotlib.pyplot as plt

def read_test_data_to_list(filename):
	with open(filename) as f:
		list = []
		for line in f:
			line = line.split(',')
			if line:
				line = [float(i) for i in line]
				list.append(line)
	# transpose b/c the matlab script used to generate these files is column-major and we want row-major
	return np.array(list).transpose()

def read_data(num):
	# read in test data
	Xtrain = read_test_data_to_list("Xtrain" + num + ".txt")
	ytrain = (read_test_data_to_list("ytrain" + num + ".txt") - 1).flatten()
	Xtest = read_test_data_to_list("Xtest" + num + ".txt")
	ytest = (read_test_data_to_list("ytest" + num + ".txt") - 1).flatten()
	return Xtrain, ytrain, Xtest, ytest
