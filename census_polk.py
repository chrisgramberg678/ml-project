import ml_project as ml
import numpy as np
import time
import pickle
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')
test_data = np.load('test_data.npy')
test_labels = np.load('test_labels.npy')

step_sizes = [1,.5,.3,.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001]
epochs = 30
data = (train_data, train_labels, test_data, test_labels)

def train_POLK(step_size, sigma, eps, epochs, data):
    train_data, train_labels, test_data, test_labels = data
    train_losses = []
    test_losses = []
    train_errors = []
    test_errors = []
    model_orders = []
    step_times = []
    BS = 10
    kernel = ml.gaussian_kernel(sigma)
    model = ml.sklr_model(kernel, 1e-9, eps)
    sgd = ml.SGD(model)
    print('*********************')
    print('training POLK with sigma={} error threshold={} step size={}. '.format(sigma, eps, step_size))
    for e in range(epochs):
        print('epoch: ', e)
        epoch_start = time.time()
        seed = e
        np.random.seed(seed)
        np.random.shuffle(train_data)
        np.random.seed(seed)
        np.random.shuffle(train_labels)
        flag = 0
        prev_size = 0
        for i in range(0, train_data.shape[0], BS):
            start = time.time()
            sgd.fit(step_size, train_data[i:i+BS], train_labels[i:i+BS])
            end = time.time()
            # add the time to compute sgd
            step_times.append(end-start)
            # calcualte training and test loss
            train_losses.append(model.loss(train_data, train_labels))
            test_losses.append(model.loss(test_data, test_labels))
            # calculate training accuracy
            predictions = model.predict(train_data) >= .5
            train_labels.shape = predictions.shape
            correct = (predictions == train_labels).sum()
            train_errors.append(1 - (correct/(train_labels.shape[0])))
            # calculate test accuracy
            predictions = model.predict(test_data) >= .5
            test_labels.shape = predictions.shape
            correct = (predictions == test_labels).sum()
            test_errors.append(1 - (correct/(test_labels.shape[0])))
            # add the current model order
            model_order = model.dictionary().shape[0]
            model_orders.append(model_order)
            # if the model is accepting all of the values we gave it epsilon is too low - terminate early
            if prev_size + BS == model_order:
                if flag > 4:
                    raise Exception('eps too low')
                print('model order: ', model_order)
                flag += 1
            prev_size = model_order

        epoch_end = time.time()
        # print('time to run epoch: {} seconds'.format(epoch_end - epoch_start))
        # print('training loss: {}. test loss: {}'.format(train_losses[-1],test_losses[-1]))
        print('model order: ',model.dictionary().shape[0])
        # print('test error: {}'.format(test_errors[-1]))
    return train_losses, test_losses, train_errors, test_errors, step_times, model_orders

census_polk_stats = {}

# we'll use 4 different values of epsilon for each step size,
# these values were found by manual exploration when sigma = .5
epsilons_map = {
    1 : np.linspace(.0025, .003, 4),
    .5 : np.linspace(6e-4, 7e-4, 4),
    .3 : np.linspace(2e-4, 5e-4, 4),
    .1 : np.linspace(2e-5, 5e-5, 4),
    .03 : np.linspace(2e-6, 2e-5, 4),
    .01 : np.linspace(2e-7, 2e-6, 4),
    .003 : np.linspace(2e-8, 2e-7, 4),
    .001 : np.linspace(2e-9, 2e-8, 4),
    .0003 : np.linspace(2e-10, 2e-9, 4),
    .0001 : np.linspace(2e-11, 2e-10, 4),
    .00003 : np.linspace(2e-12, 2e-11, 4),
    .00001 : np.linspace(2e-13, 2e-12, 4)
}

sigmas = [.5]
for sigma in sigmas:
    for step_size in step_sizes:
        epsilons = epsilons_map[step_size]
        for eps in epsilons:
            # we're expecting an exception when the model order matches the batch size several times in a row
            try:
                result = train_POLK(step_size, sigma, eps, epochs, data)
                train_losses, test_losses, train_errors, test_errors, step_times, model_orders = result
                stat = {}
                stat['train_loss'] = train_losses
                stat['test_loss'] = test_losses
                stat['train_errors'] = train_errors
                stat['test_errors'] = test_errors
                stat['step_times'] = step_times
                stat['model_order'] = model_orders
                polk_stats[eps,sigma,step_size] = stat
            except Exception:
                pass

with open('census_polk_stats.pickle','wb') as my_file:
    pickle.dump(census_polk_stats.pickle, my_file)