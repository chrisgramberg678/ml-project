#include "model.h"

// need to define the constructor for model in order for it to work with Cython
model::model(){}
model::model(bool parametric):
	_parametric(parametric)
	{}
bool model::parametric(){return _parametric;}
VectorXd model::gradient(VectorXd w, MatrixXd X, VectorXd y){return VectorXd();}
double model::loss(VectorXd w, MatrixXd X, VectorXd y){return 0;}

// **********************************************
// Implementation of linear least squares model *
// **********************************************

linear_least_squares_model::linear_least_squares_model():
	model(true)
	{}

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

binary_logistic_regression_model::binary_logistic_regression_model():
	model(true)
	{}

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
	model(false),
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
	for(int i = 0; i < X.cols(); ++i){
		VectorXd kxx_i = _k->gram_matrix(X, X.col(i));
		double e = exp(w.transpose() * (kxx_i));
		double id = (y(i) == 0 ? 1 : 0);
		result -= (((e/(e+1) - id) * kxx_i))/X.cols();
	}
	result += ((_KXX * w )* _lambda);
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
	}
	loss += (_lambda/2 * (w.transpose() * _KXX)* w);
	loss /= X.cols();
	return loss;
}

// **********************************************************************************************
// Implementation of model for stochastic descent with a logistic regression model with kernels *
// **********************************************************************************************

stochstic_kernel_logistic_regression_model::stochstic_kernel_logistic_regression_model(){}

stochstic_kernel_logistic_regression_model::stochstic_kernel_logistic_regression_model(kernel* k, double lambda):
	kernel_binary_logistic_regression_model(k, lambda)
	{}

VectorXd stochstic_kernel_logistic_regression_model::gradient(VectorXd w, VectorXd X, VectorXd y){
	double exp_f = exp(f(w, X));
	VectorXd result = (exp_f/(1 + exp_f) - (1 - y(0))) * _k->gram_matrix(_dictionary, X) + _lambda * f(w, X);
	return result;
}

double stochstic_kernel_logistic_regression_model::loss(VectorXd w, MatrixXd X, VectorXd y){
	double result = 0;
	return result;
}

double stochstic_kernel_logistic_regression_model::f(VectorXd w, VectorXd X){
	// VectorXd sum(w.rows());
	// for(int z = 0; z < w.rows(); z++){
	// 	sum(z) = 0;
	// }
	// MatrixXd kernel_mat = _k->gram_matrix(_dictionary, X);
	// for(int i = 0; i < w.rows(); ++i){
	// 	sum += w(i) * kernel_mat.col(i);
	// }
	// return sum;
	double sum = 0;
	for(int i = 0; i < w.rows(); ++i){
		sum += w(i)*_k->k(_dictionary(i),X(i));
	}
	return sum;
}