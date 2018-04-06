import numpy as np
import ml_project as ml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.datasets import make_blobs
import figures
import test_util

def scatter_by_class(features, labels, titles, models=None):
	print("plotting...")
	if features.shape[0] != labels.shape[0]:
		raise ValueError("Features and labels must have the same number of samples to be plotted")
	if len(labels.shape) == 1:
		labels.shape = labels.shape[0],1
	cat = np.concatenate((features,labels),1)
	ones = np.array([x[:2] for x in cat if x[2]==1])
	zeros = np.array([x[:2] for x in cat if x[2]==0])
	fig, axs = plt.subplots(ncols=len(models))
	if models!=None:
		if len(titles) != len(models):
			raise Error("Make sure to include a title for each model! len(titles)={}. len(models)={}".format(len(titles),len(models)))
		for title, model, axs in zip(titles, models, axs):
			ax.scatter(ones[:,0],ones[:,1],c='b')
			ax.scatter(zeros[:,0],zeros[:,1],c='r')
			ax.title(title)
			xmin, xmax = plt.xlim()
			ymin, ymax = plt.ylim()
			contour_samples = 1000.0
			x_interval = abs(xmin-xmax)/contour_samples
			y_interval = abs(ymin-ymax)/contour_samples
			xs = np.arange(xmin, xmax+x_interval, x_interval)
			ys = np.arange(ymin, ymax+y_interval, y_interval)
			xx, yy = np.meshgrid(xs, ys)
			Z = model.predict(np.c_[xx.ravel(),yy.ravel()])
			Z.shape = xx.shape
			ax.contour(xx, yy, Z)
	else:
		plt.scatter(ones[:,0],ones[:,1],c='b')
		plt.scatter(zeros[:,0],zeros[:,1],c='r')
		plt.title(titles)
	plt.show()

# create a synthetic dataset
# samples = 2000
# train_ind = int(.8 * samples)
# X, Y = make_blobs(n_samples=samples, n_features=2, centers=2, cluster_std=1.0)
# not sure why this^ doesn't classify well...

train_x, train_y, test_x, test_y = test_util.read_data("0")

# for various values of hyperparameters classify the dataset
errs = [.01, .001, .0008, .0007, .0006, .0005]
sigma = .2
# track the stats for each set of hyperparameters
stats = {}
# how much to train
epochs = 10
batch_size = 10
step_size = .3
kernel = ml.gaussian_kernel(sigma)
for err in errs:
	# for some combination of the hyperparams we need to collect some stuff every epoch
	# - the loss on the training set and the test set
	# - the model order
	# - the error rate on the test set
	train_losses = []
	test_losses = []
	test_errors = []
	model_orders = []
	model = ml.sklr_model(kernel, 1e-9, err)
	sgd = ml.SGD(model)
	print("error threshold: ", err)
	for e in range(epochs):
		print("epoch: ", e)
		for i in range(0, train_x.shape[0], batch_size):
			sgd.fit(step_size, train_x[i:i+batch_size], train_y[i:i+batch_size])
			train_losses.append(model.loss(train_x, train_y))
			test_losses.append(model.loss(test_x, test_y))
			predictions = model.predict(test_x)
			test_y.shape = predictions.shape
			correct = (predictions == test_y).sum()
			test_errors.append(1 - (correct/(test_y.shape[0])))
			model_orders.append(model.dictionary().shape[0])
		print(model.dictionary().shape[0])
	# save all of this keyed by the err
	stat = {}
	stat['train_losses'] = train_losses
	stat['test_losses'] = test_losses
	stat['test_errors'] = test_errors
	stat['model_orders'] = model_orders
	stat['model'] = model
	stats[err] = stat

# plot all of this stuff

# first plot the losses
colors = iter(cm.rainbow(np.linspace(0, 1, len(errs))))
fig, (ax1, ax2) = plt.subplots(ncols=2)
loss_key = []
for err in errs:
	stat = stats[err]
	c = next(colors)
	ax1.plot(np.arange(1,len(stat['train_losses'])+1), stat['train_losses'], linestyle='-', color=c)
	ax2.plot(np.arange(1,len(stat['test_losses'])+1), stat['test_losses'], linestyle='-', color=c)
	key = mpatches.Patch(color=c, label='Err = {}.'.format(err))
	loss_key.append(key)
ax1.set_xlabel('Training Samples * {}'.format(batch_size * epochs))
ax2.set_xlabel('Training Samples * {}'.format(batch_size * epochs))
ax1.set_ylabel('Training Loss')
ax2.set_ylabel('Test Loss')
ax1.legend(handles=loss_key)
ax2.legend(handles=loss_key)
plt.show()

# then plot the test error
colors = iter(cm.rainbow(np.linspace(0, 1, len(errs))))
fig, ax = plt.subplots()
error_key = []
for err in errs:
	stat = stats[err]
	c = next(colors)
	ax.plot(np.arange(1,len(stat['test_errors'])+1), stat['test_errors'], linestyle='-', color=c)
	error_key.append(mpatches.Patch(color=c, label='Err = {}'.format(err)))
ax.set_xlabel('Training Samples * {}'.format(batch_size))
ax.set_ylabel('Test Error')
ax.legend(handles=error_key)
plt.show()

# and the model order
colors = iter(cm.rainbow(np.linspace(0, 1, len(errs))))
fig, ax = plt.subplots()
order_key = []
for err in errs:
	stat = stats[err]
	c = next(colors)
	ax.plot(np.arange(1,len(stat['model_orders'])+1), stat['model_orders'], linestyle='-', color=c)
	order_key.append(mpatches.Patch(color=c, label='Err = {}'.format(err)))
ax.set_xlabel('Training Samples * {}'.format(batch_size))
ax.set_ylabel('Model Order')
ax.legend(handles=order_key)
plt.show()

# finally, plot the decision boundaries
models = []
titles = []
for err in errs:
	stat = stats[err]
	models.append(stat['model'])
	titles.append('Decionsion boundary using error_threshold = {}'.format(err))
scatter_by_class(test_x, test_y, titles, models)