#include "model.h"

// need to define the constructor for model in order for it to work with Cython
model::model():
	_parametric(0)
	{}
model::model(bool parametric):
	_parametric(parametric)
	{}
bool model::parametric(){return _parametric;}
VectorXd model::gradient(VectorXd w, MatrixXd X, VectorXd y){return VectorXd();}
double model::loss(VectorXd w, MatrixXd X, VectorXd y){return 0;}
VectorXd model::predict(Map<VectorXd> weights, Map<MatrixXd> X){return VectorXd::Zero(weights.size());}

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

VectorXd linear_least_squares_model::predict(Map<VectorXd> weights, Map<MatrixXd> X){
	if(weights.size() != X.rows()){
		throw invalid_argument("weights must have same size as number of rows in X. weights.size(): " + 
			to_string(weights.size()) + ". X.rows(): " + to_string(X.rows()) + ".");
	}
	return weights * X;
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

VectorXd binary_logistic_regression_model::predict(Map<VectorXd> weights, Map<MatrixXd> X){
	if(weights.size() != X.rows()){
		throw invalid_argument("weights must have same size as number of rows in X. weights.size(): " + 
			to_string(weights.size()) + ". X.rows(): " + to_string(X.rows()) + ".");
	}
	// P(y_i=1|x_i) = exp(w.transpose() * x_i)/(1 + exp(w.transpose() * x_i)) 
	int s = X.cols();
	VectorXd probabilities = VectorXd::Zero(s);
	VectorXd labels = VectorXd::Zero(s);
	for(int c = 0; c < s; ++c){
		double e = exp(weights.transpose() * X.col(c));
		probabilities(c) = e/(e+1);
		labels(c) = probabilities(c) > .5 ? 1.0 : 0.0;
	}
	return labels;
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
	loss /= X.cols();
	loss += (_lambda/2 * (w.transpose() * _KXX)* w);
	return loss;
}

VectorXd kernel_binary_logistic_regression_model::predict(Map<VectorXd> weights, Map<MatrixXd> X){
	if(weights.size() != X.rows()){
		throw invalid_argument("weights must have same size as number of rows in X. weights.size(): " + 
			to_string(weights.size()) + ". X.rows(): " + to_string(X.rows()) + ".");
	}
	return weights;
}

// **********************************************************************************************
// Implementation of model for stochastic descent with a logistic regression model with kernels *
// **********************************************************************************************

stochastic_kernel_logistic_regression_model::stochastic_kernel_logistic_regression_model(){}

stochastic_kernel_logistic_regression_model::stochastic_kernel_logistic_regression_model(kernel* k, double lambda):
	kernel_binary_logistic_regression_model(k, lambda),
	_dictionary()
	{}

VectorXd stochastic_kernel_logistic_regression_model::gradient(VectorXd w, MatrixXd X, VectorXd y){
	// determine the batch size
	double B = X.cols();
	VectorXd result = VectorXd::Zero(w.size() + 1);
	double weight = 0;
	// compute the average gradient for the batch to get the new weight
	for(int c = 0; c < B; ++c){
		if(_dictionary.size() == 0){
			_dictionary = MatrixXd(X.rows(),1);
		}
		else{
			_dictionary.conservativeResize(_dictionary.rows(), _dictionary.cols()+1);
		}
		_dictionary.col(_dictionary.cols()-1) = X.col(c);
		// compute the gradient at X 
		weight += _lambda * f(w, X);
	}
	weight = weight / B;
	// decay the weights of w by lambda
	for(int i = 0; i < w.size(); ++i){
		result(i) = _lambda * w(i);
	}
	// append the new weight
	result(result.size() - 1) = weight;
	return result;
}

double stochastic_kernel_logistic_regression_model::loss(VectorXd w, MatrixXd X, VectorXd y){
	double loss = 0;
	// ln(1 + exp(f(x))) - (1-y)*f(x) + lambda/2 * ||f||^2
	for(int c = 0; c < X.cols(); ++c){
		double f_x = f(w, X.col(c));
		loss += log(1 + f_x) - (1 - y(c)) * f_x + _lambda * f_x * f_x;
	}
	loss /= X.cols();
	return loss;
}

VectorXd stochastic_kernel_logistic_regression_model::predict(Map<VectorXd> weights, Map<MatrixXd> X){
	return VectorXd::Zero(weights.size());
}

double stochastic_kernel_logistic_regression_model::f(VectorXd w, VectorXd X){
	double result = 0;
	for(int c = 0; c < _dictionary.cols(); ++c){
		result += w(c) * _k->k(_dictionary.col(c), X);
	}
	return result;
}