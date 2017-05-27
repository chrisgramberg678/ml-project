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
	return result;
}

// sum(i=1->N){ln(1+exp(w_t*x_i)) - y_i*w_t*x_i}
double binary_logistic_regression_model::loss(VectorXd w, MatrixXd X, VectorXd y){
	double loss = 0;
	for(int i = 0; i < X.cols(); ++i){
		loss += log(1+exp(w.transpose() * X.col(i))) - (y(i) * w.transpose() * X.col(i));
	}
	return loss;
}

