import numpy as np

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
	Xtrain = read_test_data_to_list("tests/Xtrain" + num + ".txt")
	ytrain = (read_test_data_to_list("tests/ytrain" + num + ".txt") - 1).flatten()
	Xtest = read_test_data_to_list("tests/Xtest" + num + ".txt")
	ytest = (read_test_data_to_list("tests/ytest" + num + ".txt") - 1).flatten()
	return Xtrain, ytrain, Xtest, ytest

def plot_by_class(X, y, n):
	n = min(n,y.size)
	for i in range(n):
		if y[i] == 1:
			plt.scatter(X[0,i],X[1,i], c='b')
		elif y[i] == 0:
			plt.scatter(X[0,i],X[1,i], c='r')