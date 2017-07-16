#include "model.h"

// need to define the constructor for model in order for it to work with Cython
model::model(){}
VectorXd model::gradient(VectorXd w, MatrixXd X, VectorXd y){return VectorXd();}
double model::loss(VectorXd w, MatrixXd X, VectorXd y){return 0;}

// **********************************************
// Implementation of linear least squares model *
// **********************************************

linear_least_squares_model::linear_least_squares_model(){}

VectorXd linear_least_squares_model::gradient(VectorXd w, MatrixXd X, VectorXd y){
	VectorXd result(w.rows());
	for(int i = 0; i < X.cols(); ++i){
		result += 2*(X.col(i) * ((w.transpose() * X.col(i)) - y(i)))/X.cols();
	}
	return result;
}

// L(a,b) = 1/N sum (y-ax-b)^2
double linear_least_squares_model::loss(VectorXd w, MatrixXd X, VectorXd y){
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		double temp = w.transpose() * X.col(i) - y(i);
		loss += temp*temp;
	}
	loss /= X.cols();
	return loss;
}

// *********************************************
// Implementation of logistic regression model *
// *********************************************

binary_logistic_regression_model::binary_logistic_regression_model(){}

// sum(i=1->N){x_i * (Pr(y=1|x_i) - y_1)}
VectorXd binary_logistic_regression_model::gradient(VectorXd w, MatrixXd X, VectorXd y){
	VectorXd result(w.rows());
	for(int i = 0; i < X.cols(); ++i){
		result += (( exp(w.transpose() * X.col(i)) / (1 + exp(w.transpose() * X.col(i)))) - y(i)) * X.col(i);
	}
	result /=X.cols();
	return result;
}

// sum(i=1->N){ln(1+exp(w_t*x_i)) - y_i*w_t*x_i}
double binary_logistic_regression_model::loss(VectorXd w, MatrixXd X, VectorXd y){
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		loss += log(1+exp(w.transpose() * X.col(i))) - (y(i) * w.transpose() * X.col(i));
	}
	loss /=X.cols();
	return loss;
}

// **********************************************************
// Implementation of logistic regression model with kernels *
// **********************************************************

kernel_binary_logistic_regression_model::kernel_binary_logistic_regression_model(){}

kernel_binary_logistic_regression_model::kernel_binary_logistic_regression_model(kernel* k, double lambda):
	_lambda(lambda),
	_k(k),
	first(true)
	{}

VectorXd kernel_binary_logistic_regression_model::gradient(VectorXd w, MatrixXd X, VectorXd y){
	if(first){
		_KXX = _k->gram_matrix(X, X);
		first = false;
	}
	VectorXd result(w.rows());
	for(int j = 0; j < w.rows(); ++j){
		result(j) = 0;
	}
	cout << _KXX.rows() << "," << _KXX.cols() << "," << w.rows() << endl;
	for(int i = 0; i < X.cols(); ++i){
		VectorXd kxx_i = _k->gram_matrix(X, X.col(i));
		double e = exp(w.transpose() * (kxx_i));
		double id = (y(i) == 0 ? 1 : 0);
		result -= (((e/(e+1) - id) * kxx_i) + ((_KXX * w )* _lambda))/X.cols();
	}
	// this right here is what I'm confused about.
	cout << result.rows() << "," << w.rows() << endl; 
	return result;
}

double kernel_binary_logistic_regression_model::loss(VectorXd w, MatrixXd X, VectorXd y){
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		VectorXd kxx_i = _k->gram_matrix(X, X.col(i));
		double e = exp(w.transpose() * kxx_i);
		double id = (y(i) == 0 ? 1 : 0);
		loss -= log(1 + e);
		double temp = w.transpose() * kxx_i;
		loss -= temp * id;
		loss += (_lambda/2 * (w.transpose() * _KXX)* w);
	}
	loss /= X.cols();
	return loss;
}