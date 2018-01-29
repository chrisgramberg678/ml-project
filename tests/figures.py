import matplotlib.pyplot as plt
import numpy as np
import ml_project as ml
import test_util

def scatter_by_class(features, labels, model=None):
	print("plotting...")
	if features.shape[0] != labels.shape[0]:
		raise ValueError("Features and labels must have the same number of samples to be plotted")
	if len(labels.shape) == 1:
		labels.shape = labels.shape[0],1
	cat = np.concatenate((features,labels),1)
	ones = np.array([x[:2] for x in cat if x[2]==1])
	zeros = np.array([x[:2] for x in cat if x[2]==0])
	plt.scatter(ones[:,0],ones[:,1],c='b')
	plt.scatter(zeros[:,0],zeros[:,1],c='r')
	xmin, xmax = plt.xlim()
	ymin, ymax = plt.ylim()
	contour_samples = 1000.0
	x_interval = abs(xmin-xmax)/contour_samples
	y_interval = abs(ymin-ymax)/contour_samples
	xs = np.arange(xmin, xmax+x_interval, x_interval)
	ys = np.arange(ymin, ymax+y_interval, y_interval)
	xx, yy = np.meshgrid(xs, ys)
	if model!=None:
		Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
		print(Z)
		Z.shape = xx.shape
		plt.contour(xx, yy, Z, levels=[0])
	plt.show()

N = 400
xtrain, ytrain, xtest, ytest = test_util.read_data("0")
# logistic regression model, trained in batch
m = ml.blr_model()
solver = ml.BGD(xtrain[:N], ytrain[:N], m)
solver.fit(np.random.rand(2,1), step_size=.001, conv_type="step_precision", conv_val=.000001)
scatter_by_class(xtrain[:N], ytrain[:N], m)
# kernel logistic regression model, trained online
k = ml.gaussian_kernel(.1)
m = ml.sklr_model(k, 0)
solver = ml.SGD(m)
print("solving...")
epochs = 20
batch_size = 10
weights = np.random.rand(1,1)
for e in range(epochs):
	print("epoch",e)
	for i in range(0,N,batch_size):
		weights = solver.fit(weights, .3, xtrain[i:i+batch_size], ytrain[i:i+batch_size])
scatter_by_class(xtrain[:N], ytrain[:N], m)
