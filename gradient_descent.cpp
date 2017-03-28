/*Impementation of gradient_descent*/

#include "gradient_descent.h"

// computes the gradient of some vector w in terms of the training data
VectorXd gradient_descent::gradient(VectorXd w){
	VectorXd result;
	for(int i = 0; i < _X.cols(); ++i){
		result += 2*(_X.col(i) * ((w.transpose() * _X.col(i)) - _y(i)))/w.cols();
	}
	return result;
}

// determines if the values in w are beneath the precision value
bool gradient_descent::done(VectorXd w, double precision){
	// convert w to an Array so we can do coefficent-wise ops like abs()
	ArrayXd temp = w.array();
	temp = temp.abs();
	return (temp < precision).all();
}

// L(a,b) = 1/N sum (y-ax-b)^2
// we know this is working when the value of the loss function gets smaller
double gradient_descent::Loss(VectorXd w){
	double loss = 0;
	for(int i = 0; i < _X.cols(); ++i){
		double temp = w.transpose() * _X.col(i) - _y(i);
		loss += temp*temp;
	}
	loss /= _X.cols();
	return loss;
}

// default constructor to appease Cython
gradient_descent::gradient_descent(){}

// construction using arbitrary size data
gradient_descent::gradient_descent(MatrixXd X, VectorXd y):
	_X(X),
	_y(y)
	{}
	
// does the actual fitting using gradient descent
// params: 
VectorXd gradient_descent::fit(VectorXd init, double gamma, double precision, bool verbose){
	assert(init.rows() == _X.rows());
	// these are used for iteration
	// think of w_k-1 and w_k where diff is the difference btw the two
	VectorXd w_k1, w_k, diff(init.rows());
	// starting values for w the weights we're solving for
	w_k = init;
	// fill diff with ones so we can get false from done
	for(int i = 0; i < w_k.rows(); ++i){
		diff(i) = 1;
	}
	double loss = 10000000;
	int i = 0;
	while( !done(diff, precision) ){
		if(verbose){
   			cout << "iteration: " << i++ << endl;
			cout << "Loss: " << loss << endl;
			cout << "gradient: " << w_k << endl;
		}
		w_k1 = w_k;
		w_k -= gamma*gradient(w_k1);
		diff = w_k1 - w_k;
		loss = Loss(w_k);
		if(loss == numeric_limits<double>::infinity()){
			throw runtime_error("we have diverged!");
		}
	}
	if(verbose){
		cout << "coefficients: " << endl << w_k << endl;
	}
	return w_k;
}

// a slightly different version of fit so that I don't have 
// to wrap Eigen for Cython
vector<double> gradient_descent::py_fit(vector<int> ab, double gamma, double precision, bool verbose){
	vector<double> r;
	return r;
}