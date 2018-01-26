#include "model.h"

// need to define the constructor for model in order for it to work with Cython
model::model():
	_parametric(0)
	{}
model::model(bool parametric):
	_parametric(parametric)
	{}
bool model::parametric(){return _parametric;}
VectorXd model::gradient(MatrixXd X, VectorXd y){return VectorXd();}
double model::loss(MatrixXd X, VectorXd y){return 0;}
VectorXd model::predict(Map<MatrixXd> X){return VectorXd::Zero(X.rows());}
void model::init_weights(int num_weights){
	_weights = VectorXd::Zero(num_weights);
	// taken directly from the http://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
	// TODO: consider allowing the model weights to be initialized in different ways
	random_device rd;  
    mt19937 gen(rd()); 
    uniform_real_distribution<> dis(0.0, 1.0);
	for(int i = 0; i < num_weights; ++i){
		_weights(i) = dis(gen);
	}
}
void model::init_weights(VectorXd init){
	_weights = init;
}

// **********************************************
// Implementation of linear least squares model *
// **********************************************

linear_least_squares_model::linear_least_squares_model():
	model(true)
	{}

VectorXd linear_least_squares_model::gradient(MatrixXd X, VectorXd y){
	VectorXd result(_weights.rows());
	for(int i = 0; i < X.cols(); ++i){
		result += 2*(X.col(i) * ((_weights.transpose() * X.col(i)) - y(i)))/X.cols();
	}
	return result;
}

// L(a,b) = 1/N sum (y-ax-b)^2
double linear_least_squares_model::loss(MatrixXd X, VectorXd y){
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		double temp = _weights.transpose() * X.col(i) - y(i);
		loss += temp*temp;
	}
	loss /= X.cols();
	return loss;
}

VectorXd linear_least_squares_model::predict(Map<MatrixXd> X){
	if(_weights.size() != X.rows()){
		throw invalid_argument("weights must have same size as number of rows in X. weights.size(): " + 
			to_string(_weights.size()) + ". X.rows(): " + to_string(X.rows()) + ".");
	}
	return _weights.transpose() * X;
}

// *********************************************
// Implementation of logistic regression model *
// *********************************************

binary_logistic_regression_model::binary_logistic_regression_model():
	model(true)
	{}

// sum(i=1->N){x_i * (Pr(y=1|x_i) - y_1)}
VectorXd binary_logistic_regression_model::gradient(MatrixXd X, VectorXd y){
	VectorXd result(_weights.rows());
	for(int i = 0; i < X.cols(); ++i){
		result += (( exp(_weights.transpose() * X.col(i)) / (1 + exp(_weights.transpose() * X.col(i)))) - y(i)) * X.col(i);
	}
	result /=X.cols();
	return result;
}

// sum(i=1->N){ln(1+exp(w_t*x_i)) - y_i*w_t*x_i}
double binary_logistic_regression_model::loss(MatrixXd X, VectorXd y){
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		loss += log(1+exp(_weights.transpose() * X.col(i))) - (y(i) * _weights.transpose() * X.col(i));
	}
	loss /=X.cols();
	return loss;
}

VectorXd binary_logistic_regression_model::predict(Map<MatrixXd> X){
	if(_weights.size() != X.rows()){
		throw invalid_argument("weights must have same size as number of rows in X. weights.size(): " + 
			to_string(_weights.size()) + ". X.rows(): " + to_string(X.rows()) + ".");
	}
	// P(y_i=1|x_i) = exp(w.transpose() * x_i)/(1 + exp(w.transpose() * x_i)) 
	int s = X.cols();
	VectorXd probabilities = VectorXd::Zero(s);
	VectorXd labels = VectorXd::Zero(s);
	for(int c = 0; c < s; ++c){
		double e = exp(_weights.transpose() * X.col(c));
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

VectorXd kernel_binary_logistic_regression_model::gradient(MatrixXd X, VectorXd y){
	if(first){
		_X_train = X;
		_KXX = _k->gram_matrix(X, X);
		first = false;
	}
	VectorXd result = VectorXd::Zero(_weights.rows());
	for(int i = 0; i < X.cols(); ++i){
		VectorXd kxx_i = _k->gram_matrix(X, X.col(i));
		double e = exp(_weights.transpose() * (kxx_i));
		double id = (y(i) == 0 ? 1 : 0);
		result -= (((e/(e+1) - id) * kxx_i))/X.cols();
	}
	result += ((_KXX * _weights )* _lambda);
	return result;
}

double kernel_binary_logistic_regression_model::loss(MatrixXd X, VectorXd y){
	if(first){
		_X_train = X;
		_KXX = _k->gram_matrix(X, X);
		first = false;
	}
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		VectorXd kxx_i = _k->gram_matrix(X, X.col(i));
		double e = exp(_weights.transpose() * kxx_i);
		double id = (y(i) == 0 ? 1 : 0);
		loss -= log(1 + e);
		double temp = _weights.transpose() * kxx_i;
		loss -= temp * id;
	}
	loss /= X.cols();
	loss += (_lambda/2 * (_weights.transpose() * _KXX)* _weights);
	return loss;
}

VectorXd kernel_binary_logistic_regression_model::predict(Map<MatrixXd> X){
	// P(y_i=1|x_i) = exp(w.transpose() * kxx_i)/(1 + exp(w.transpose() * kxx_i))
	int s = X.cols();
	VectorXd probabilities = VectorXd::Zero(s);
	VectorXd labels = VectorXd::Zero(s);
	for(int c = 0; c < s; ++c){
		VectorXd kxx_i = _k->gram_matrix(_X_train, X.col(c));
		double e = exp(_weights.transpose() * (kxx_i));
		probabilities(c) = e/(e+1);
		labels(c) = probabilities(c) > .5 ? 1.0 : 0.0;
	}
	return labels;
}

// **********************************************************************************************
// Implementation of model for stochastic descent with a logistic regression model with kernels *
// **********************************************************************************************

stochastic_kernel_logistic_regression_model::stochastic_kernel_logistic_regression_model(){}

stochastic_kernel_logistic_regression_model::stochastic_kernel_logistic_regression_model(kernel* k, double lambda):
	kernel_binary_logistic_regression_model(k, lambda),
	_dictionary(),
	_KDD()
	{}

VectorXd stochastic_kernel_logistic_regression_model::gradient(MatrixXd X, VectorXd y){
	// the batch size
	int B = X.cols();
	// determine the batch size, for each sample in the batch we'll add a new weight
	VectorXd result = VectorXd::Zero(_weights.size() + B);
	// decay the old weights by lambda, before adding new weights
	for(int i = 0; i < _weights.size(); ++i){
		result(i) = _lambda * _weights(i);
	}
	// add the new weights to the result
	for(int b = 0; b < B; ++b){
		// compute the new weight
		// w_m+1 = eta * (P(y=0|x) - (1-y))
		// note that the eta refers to the gradient step size and will be handled by the solver
		double exp_fx = exp(f(X.col(b)));
		double weight = (exp_fx/(1+exp_fx)) - (1 - y(b));
		// append the new weight to the result
		result(_weights.size() + b) = weight/B;
	}
	update_dictionary(X);
	return result;
}

void stochastic_kernel_logistic_regression_model::update_dictionary(MatrixXd X){
	// the batch size
	int B = X.cols();
	int old_size = _dictionary.cols();
	// make room in the dictionary for the new samples
	if(_dictionary.cols() == 0){
		_dictionary = MatrixXd(X.rows(),B);
	}
	else{
		_dictionary.conservativeResize(_dictionary.rows(), _dictionary.cols()+B);
	}
	// update the dictionary after computing the new weights
	for(int b = 0; b < B; ++b){
		_dictionary.col(old_size + b) = X.col(b);
	}
}
void stochastic_kernel_logistic_regression_model::update_KDD(){
	if(_KDD.size() == 0){
		_KDD = _k->gram_matrix(_dictionary,_dictionary);
	}
	// instead of recalculating the entire matrix just copy the old parts and compute the new
	else{
		int diff = _dictionary.cols() - _KDD.cols();
		_KDD.conservativeResize(_KDD.rows() + diff, _KDD.cols() + diff);
		for(int i = diff; i < _dictionary.cols(); ++i){
			for(int j = 0; j < i; ++j){
				double temp = _k->k(_dictionary.col(j), _dictionary.col(i));
				_KDD(i,j) = temp;
				_KDD(j,i) = temp;
			}
			_KDD(i,i) = 1;
		}
	}
}


double stochastic_kernel_logistic_regression_model::loss(MatrixXd X, VectorXd y){
	// ln(1 + exp(f(x))) - (1-y)*f(x) + lambda/2 * ||f||^2
	double loss = 0;
	for(int c = 0; c < X.cols(); ++c){
		double f_x = f(X.col(c));
		loss += log(1 + exp(f_x)) - ((1 - y(c)) * f_x);
	}
	loss /= X.cols();
	if(_weights.size() > 0){
		// update the regularization factor
		update_KDD();
		loss += _lambda/2 * (_weights.transpose() * _KDD) * _weights;
	}		
	return loss;
}

VectorXd stochastic_kernel_logistic_regression_model::predict(Map<MatrixXd> X){
	// P(y_i=z|x_i) = [exp(f(x_i))^(1-z)]/[1+exp(f(x_i))]
	// P(y_i=1|x_i) = 1/(1 + exp(f(x)))
	int s = X.cols();
	VectorXd probabilities = VectorXd::Zero(s);
	VectorXd labels = VectorXd::Zero(s);
	for(int c = 0; c < s; ++c){
		double e = exp(f(X.col(c)));
		probabilities(c) = 1/(e+1);
		labels(c) = probabilities(c) > .5 ? 1.0 : 0.0;
	}
	// cout << "probabilities:" << probabilities << endl;
	// cout << "labels:" << labels << endl;
	return labels;
}

double stochastic_kernel_logistic_regression_model::f(VectorXd x){
	double result = 0;
	for(int c = 0; c < _dictionary.cols(); ++c){
		result += _weights(c) * _k->k(_dictionary.col(c), x);
	}
	return result;
}