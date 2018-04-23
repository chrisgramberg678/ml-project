import numpy as np
import pickle
import ml_project as ml
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from sklearn.datasets import make_blobs
import figures
import test_util

# creates a data set with two features
def generate_dataset(classes=2, radius=5, num_samples=2000, second_samples=5, std1=.1, std2=1):
	print('getnerating data')
	class_centers = []
	# generate a cluster center for each class
	for class_center in range(classes):
		theta = np.random.rand() * 2 * np.pi
		class_centers.append(radius * np.array([np.cos(theta), np.sin(theta)]))
	
	# for each cluster center sample second_sample centers to draw the final samples from
	# store them in a dict indexed by class
	second_centers = {}
	for c in range(classes):
		second_centers[c] = np.random.normal(loc=class_centers[c], scale=[std1,std1], size=(second_samples,2))

	# for each sample, randomly assign it to a cluster
	labels = np.random.randint(classes, size=num_samples)
	# now generate the features based on a randomly selected second_sample
	samples = np.zeros((num_samples,2))
	for ind, label in enumerate(labels):
		mean = second_centers[label][np.random.randint(second_samples)]
		samples[ind] = np.random.normal(loc=mean, scale=[std2,std2])
	labels.shape = (num_samples,1)
	return samples, labels

def scatter_by_class(features, labels, titles, models=None):
	print('plotting...')
	if features.shape[0] != labels.shape[0]:
		raise ValueError('Features and labels must have the same number of samples to be plotted')
	if len(labels.shape) == 1:
		labels.shape = labels.shape[0],1
	cat = np.concatenate((features,labels),1)
	ones = np.array([x[:2] for x in cat if x[2]==1])
	zeros = np.array([x[:2] for x in cat if x[2]==0])
	if models!=None:
		if len(titles) != len(models):
			raise Error('Make sure to include a title for each model! len(titles)={}. len(models)={}'.format(len(titles),len(models)))
		for title, model in zip(titles, models):
			fig, ax = plt.subplots()
			ax.scatter(ones[:,0],ones[:,1],c='b', facecolors='none')
			ax.scatter(zeros[:,0],zeros[:,1],c='r', facecolors='none')
			dictionary = model.dictionary()
			ax.scatter(dictionary[:,0], dictionary[:,1], c='black', marker='x')
			ax.set_title(title)
			xmin, xmax = plt.xlim()
			ymin, ymax = plt.ylim()
			contour_samples = 100
			x_interval = abs(xmin-xmax)/contour_samples
			y_interval = abs(ymin-ymax)/contour_samples
			xs = np.arange(xmin, xmax+x_interval, x_interval)
			ys = np.arange(ymin, ymax+y_interval, y_interval)
			xx, yy = np.meshgrid(xs, ys)
			grid_coords = np.c_[xx.ravel(),yy.ravel()]
			Z = model.predict(grid_coords)
			Z.shape = xx.shape
			ax.contour(xx, yy, Z, 10)
			# print(grid_ones.shape)
			# print(grid_ones)
			# print('****************************')
			# print(grid_zeros.shape)
			# print(grid_zeros)
			ax.scatter(xx[np.where(Z>.5)], yy[np.where(Z>.5)], marker=',', s=1, c='b', alpha=.5)
			ax.scatter(xx[np.where(Z<=.5)], yy[np.where(Z<=.5)], marker=',', s=1, c='r', alpha=.5)
			plt.grid()
			plt.show()
	else:
		plt.scatter(ones[:,0], ones[:,1], c='b', marker='.', facecolors='none')
		plt.scatter(zeros[:,0], zeros[:,1],c='r', marker='.', facecolors='none')
		# xx_ones, yy_ones = 
		plt.title(titles)
		plt.grid()
		plt.show()

# create a synthetic dataset
# train_x, train_y, test_x, test_y = test_util.read_data('0')
samples = 2000
split_ind = int(.9*samples)
data = np.load('data.npy')
X, Y = data[:,:2], data[:,2]
train_x, train_y = X[:split_ind], Y[:split_ind]
test_x, test_y = X[split_ind:], Y[split_ind:]

# for various values of hyperparameters classify the dataset
errs = [.01, .001, .0008, .0007, .0006]
sigma = .2
# track the stats for each set of hyperparameters
stats = {}
# how much to train
epochs = 10
batch_size = 1
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
	print('*********************')
	print('error threshold: ', err)
	for e in range(epochs):
		print('epoch: ', e)
		for i in range(0, train_x.shape[0], batch_size):
			sgd.fit(step_size, train_x[i:i+batch_size], train_y[i:i+batch_size])
			train_losses.append(model.loss(train_x, train_y))
			test_losses.append(model.loss(test_x, test_y))
			predictions = model.predict(test_x) >= .5
			test_y.shape = predictions.shape
			correct = (predictions == test_y).sum()
			test_errors.append(1 - (correct/(test_y.shape[0])))
			model_orders.append(model.dictionary().shape[0])
		# print('test acc: {}/{}'.format(correct,test_y.shape[0]))
		print('model order: ',model.dictionary().shape[0])
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
	key = mpatches.Patch(color=c, label='$\epsilon$ = {}.'.format(err))
	loss_key.append(key)
ax1.set_xlabel('Training Samples * {}'.format(batch_size * epochs))
ax2.set_xlabel('Training Samples * {}'.format(batch_size * epochs))
ax1.set_ylabel('Training Loss')
ax2.set_ylabel('Test Loss')
ax1.legend(handles=loss_key)
ax2.legend(handles=loss_key)
ax1.grid()
ax2.grid()
plt.show()

# then plot the test error
colors = iter(cm.rainbow(np.linspace(0, 1, len(errs))))
fig, ax = plt.subplots()
error_key = []
for err in errs:
	stat = stats[err]
	c = next(colors)
	ax.plot(np.arange(1,len(stat['test_errors'])+1), stat['test_errors'], linestyle='-', color=c)
	error_key.append(mpatches.Patch(color=c, label=' = {}'.format(err)))
ax.set_xlabel('Training Samples * {}'.format(batch_size))
ax.set_ylabel('Test Error')
ax.legend(handles=error_key)
plt.grid()
plt.show()

# and the model order
colors = iter(cm.rainbow(np.linspace(0, 1, len(errs))))
fig, ax = plt.subplots()
order_key = []
for err in errs:
	stat = stats[err]
	c = next(colors)
	ax.plot(np.arange(1,len(stat['model_orders'])+1), stat['model_orders'], linestyle='-', color=c)
	order_key.append(mpatches.Patch(color=c, label='$\epsilon$ = {}'.format(err)))
ax.set_xlabel('Training Samples * {}'.format(batch_size))
ax.set_ylabel('Model Order')
ax.legend(handles=order_key)
plt.grid()
plt.show()

# finally, plot the decision boundaries
models = []
titles = []
for err in errs:
	stat = stats[err]
	models.append(stat['model'])
	titles.append('Decionsion boundary using error_threshold = {}'.format(err))
scatter_by_class(test_x, test_y, titles, models)