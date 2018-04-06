#include "model.h"

// need to define the constructor for model in order for it to work with Cython
model::model():
	_parametric(0)
	{}

model::model(bool parametric):
	_parametric(parametric)
	{}

bool model::parametric(){return _parametric;}

void model::init_weights(int num_weights){
	_weights = VectorXd(num_weights);
	std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd{0,5};
	for(int i = 0; i < num_weights; ++i){
			_weights(i) = nd(gen);
	}
}

void model::update_weights(VectorXd new_w){
	_weights = new_w;
}

void model::update_weights(VectorXd new_w, MatrixXd X){
	_weights = new_w;
}

VectorXd model::get_weights(){
	return _weights;
}

double model::loss(Map<MatrixXd> X, Map<VectorXd> y){
	this->loss(MatrixXd(X),VectorXd(y));
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
	for(int c = 0; c < s; ++c){
		double e = exp(_weights.transpose() * X.col(c));
		probabilities(c) = e/(e+1);
	}
	return probabilities;
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
		loss += log(1+exp(_weights.transpose() * _KXX.col(i)) + 1e-3) - (y(i) * _weights.transpose() * _KXX.col(i));
	}
	loss /= X.cols();
	loss += (_lambda/2 * (_weights.transpose() * _KXX)* _weights);
	return loss;
}

VectorXd kernel_binary_logistic_regression_model::predict(Map<MatrixXd> X){
	// P(y_i=1|x_i) = exp(w.transpose() * kxx_i)/(1 + exp(w.transpose() * kxx_i))
	int s = X.cols();
	VectorXd probabilities = VectorXd::Zero(s);
	for(int c = 0; c < s; ++c){
		VectorXd kxx_i = _k->gram_matrix(_X_train, X.col(c));
		double e = exp(_weights.transpose() * (kxx_i));
		probabilities(c) = e/(e+1);
	}
	return probabilities;
}

// **********************************************************************************************
// Implementation of model for stochastic descent with a logistic regression model with kernels *
// **********************************************************************************************

stochastic_kernel_logistic_regression_model::stochastic_kernel_logistic_regression_model(){}

stochastic_kernel_logistic_regression_model::stochastic_kernel_logistic_regression_model(kernel* k, double lambda, double err_max):
	kernel_binary_logistic_regression_model(k, lambda),
	_err_max(err_max),
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
	return result;
}

void stochastic_kernel_logistic_regression_model::update_weights(VectorXd new_w, MatrixXd X){
	// update the weights
	_weights = new_w;
	// update the dictionary
	update_dictionary(X);
	// update KDD and KDD^-1
	update_KDD();
	update_KDD_inverse();
	// run omp to prune the dictionary
	prune_dictionary();
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
		_KDD = _k->gram_matrix_stable(_dictionary,_dictionary);
	}
	// instead of recalculating the entire matrix just copy the old parts and compute the new
	else{
		int i = _KDD.cols();
		_KDD.conservativeResize(_dictionary.cols(), _dictionary.cols());
		for(; i < _dictionary.cols(); ++i){
			VectorXd new_col = _k->gram_matrix_stable(_dictionary, _dictionary.col(i));
			_KDD.col(i) = new_col;
			_KDD.row(i) = new_col.transpose();
		}
	}
}

void stochastic_kernel_logistic_regression_model::update_KDD_inverse(){
	// we'll update the iverse matrix iteratively for each new dictionary value 
	while(_KDD_inverse.cols() < _KDD.cols()){
		int c = _KDD_inverse.cols();
		// the one column version of the inverse is a 1x1 matrix with the value (K_DD(0,0))^-1.
		if(c == 0){
			_KDD_inverse = MatrixXd(1,1);
			_KDD_inverse(0,0) = 1.0/_KDD(0,0);
		}
		// add a single row and column
		else{
			// create some intermediary values
			// cout << "cols vs sample\n";
			VectorXd u1 = _k->gram_matrix_stable(_dictionary.leftCols(c), _dictionary.col(c));
			VectorXd u2 = _KDD_inverse * u1;
			// cout << "sample vs sample\n";
			double d = 1/(_k->gram_matrix_stable(_dictionary.col(c), _dictionary.col(c)).value() - (u1.transpose() * u2).value());
			VectorXd u3 = d * u2;
			MatrixXd top_left = _KDD_inverse + d * u2 * u2.transpose();
			// create _KDD_inverse from the intermediary values
			MatrixXd new_KDD_inverse = MatrixXd(c + 1, c + 1);
			new_KDD_inverse.topLeftCorner(c, c) = top_left;
			new_KDD_inverse.topRightCorner(c, 1) = -1 * u3;
			new_KDD_inverse.bottomLeftCorner(1, c) = -1 * u3.transpose();
			new_KDD_inverse(c, c) = d;
			_KDD_inverse = new_KDD_inverse;
		}
	}
}

MatrixXd stochastic_kernel_logistic_regression_model::remove_col_from_dict(MatrixXd d, int i){
	if(i > d.cols() - 1 || i < 0){
		stringstream ss;
		ss << "Cannot remove column " << i << " from dictionary of length " << d.cols() << ".";
		throw std::invalid_argument(ss.str());
	}
	// permute the target column to the end
	for(int j = i + 1; j < d.cols(); ++j){
		d.col(j - 1) = d.col(j);
	}
	// return all but the last column
	return d.leftCols(d.cols() - 1);
}

MatrixXd stochastic_kernel_logistic_regression_model::remove_sample_from_Kdd(MatrixXd Kdd, int i){
	if(i > Kdd.cols() - 1 || i < 0){
		stringstream ss;
		ss << "Cannot remove sample " << i << " from kernel matrix with dims [" << Kdd.rows() << ", " << Kdd.cols() << "].";
		throw std::invalid_argument(ss.str());
	}
	for(int j = i + 1; j < Kdd.cols(); ++j){
		Kdd.row(j - 1) = Kdd.row(j);
		Kdd.col(j - 1) = Kdd.col(j);
	}
	return Kdd.topLeftCorner(Kdd.rows() - 1, Kdd.cols() -1);
}

MatrixXd stochastic_kernel_logistic_regression_model::remove_sample_from_inverse(MatrixXd old_inverse, int i){
	if(old_inverse.rows() != old_inverse.cols()){
		stringstream ss;
		ss << "Invalid inverse matrix. Number of rows and columns should match. Given: rows = " << old_inverse.rows() << ", cols = " << old_inverse.cols();
		throw std::invalid_argument(ss.str());
	}
	if(i > old_inverse.cols() - 1 || i < 0){
		stringstream ss;
		ss << "Cannot remove column " << i << " from inverse matrix with " << old_inverse.cols() << " columns.";
		throw std::invalid_argument(ss.str());
	}
	// permute the ith column and ith column to the last column and last row
	VectorXd ith_col = old_inverse.col(i);
	double ith_col_i = ith_col(i);
	for(int j = i + 1; j < old_inverse.cols(); ++j){
		old_inverse.row(j - 1) = old_inverse.row(j);
		old_inverse.col(j - 1) = old_inverse.col(j);
		ith_col(j - 1) = ith_col(j);
	}
	ith_col(ith_col.size() - 1) = ith_col_i;
	old_inverse.bottomRows(1) = ith_col.transpose();
	old_inverse.rightCols(1) = ith_col;
	// create some intermediary values
	MatrixXd top_left = old_inverse.topLeftCorner(old_inverse.rows() - 1, old_inverse.cols() - 1);
	double d = old_inverse(old_inverse.rows() - 1, old_inverse.rows() - 1);
	VectorXd u3 = -1 * old_inverse.topRightCorner(old_inverse.rows() - 1, 1);
	VectorXd u2 = u3/d;
	// build the new inverse out of those values
	MatrixXd new_inverse = top_left - (d * (u2 * u2.transpose()));
	return new_inverse;
}

void stochastic_kernel_logistic_regression_model::prune_dictionary(){
	while(_dictionary.cols() > 1){
	// for(int j = 0; j < _dictionary.cols(); ++j){
		map<int, double> err_map;
		map<int, VectorXd> beta_map;
		// compute the error if each value was excluded
		// cout << "**************\nj = " << j << "\nd:\n" << _dictionary << "\nw:\n" << _weights << endl; 
		// cout << "**************\nd.cols() = " << _dictionary.cols() << "\nd:\n" << _dictionary << "\nw:\n" << _weights << endl; 
		for(int i = 0; i < _dictionary.cols(); i++){
			// cout << "\ni: " << i << endl;
			// remove the ith value from the dictionary and recompute the kernel matrix and its inverse
			MatrixXd d_temp = remove_col_from_dict(_dictionary, i);
			MatrixXd _KDD_temp = remove_sample_from_Kdd(_KDD, i);
			MatrixXd _KDD_inverse_temp = remove_sample_from_inverse(_KDD_inverse, i);
			MatrixXd _KDd_temp = _k->gram_matrix(_dictionary, d_temp);
			VectorXd beta = (_weights.transpose() * _KDd_temp * _KDD_inverse_temp);
			double residual_error = (_weights.transpose() * _KDD * _weights).value() + 
									(beta.transpose() * _KDD_temp * beta).value() - 
									2 * (_weights.transpose() * _KDd_temp * beta).value();
			// cout << "error = " << first_term << " + " << second_term << " - " <<  third_term << endl;
			// cout << "d_temp:\n" << d_temp << endl;
			// cout << "beta:\n" << beta << endl;
			// cout << "residual_error: " << residual_error << endl;
			err_map.insert({i, residual_error});
			beta_map.insert({i, beta});
		}
		// find the i that gave the smallest error
		double err_best = 100000;
		int best_i = -1;
		for(int i = 0; i < _dictionary.cols(); i++){
			auto err_search = err_map.find(i);
			if(err_search->second < err_best){
				best_i = err_search->first;
				err_best = err_search->second;
			}
		}
		if (err_best > _err_max){
			// cout << "err_best: " << err_best << " not better than err_max: " << _err_max << endl;
			break;
		}
		// cout << "best_i: " << best_i << endl;
		// cout << "err_best: " << err_best << endl;
		// update dictionary
		_dictionary = remove_col_from_dict(_dictionary, best_i);
		// update weights
		auto beta_search = beta_map.find(best_i);
		_weights = beta_search->second;
		// update kernel matrix and inverse
		_KDD = remove_sample_from_Kdd(_KDD, best_i);
		_KDD_inverse = remove_sample_from_inverse(_KDD_inverse, best_i);
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
	for(int c = 0; c < s; ++c){
		double e = exp(f(X.col(c)));
		probabilities(c) = 1/(e+1);
	}
	return probabilities;
}

double stochastic_kernel_logistic_regression_model::f(VectorXd x){
	double result = 0;
	for(int c = 0; c < _dictionary.cols(); ++c){
		result += _weights(c) * _k->k(_dictionary.col(c), x);
	}
	return result;
}

MatrixXd stochastic_kernel_logistic_regression_model::dictionary(){
	return _dictionary;
}