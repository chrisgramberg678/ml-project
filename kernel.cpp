/* implementation of various kernels and the gram_matrix method in the parnet kernel class  */

#include "kernel.h"

/*
 Implementation for the gram_matrix method that all children of kernel use
 It is dependent on the virtual method k() which computes the kernel between two vectors 
*/

/* 
 * X is a dxM matrix and Y is a dxN matrix
 * this function computes the Gram Matrix/Kernel Matrix K(X,Y) where the (i,j) entry of K
 * is k(x_i,y_j) and x_i, y_j are the ith and jth columns of X and Y. 
 */
MatrixXd kernel::gram_matrix(MatrixXd X, MatrixXd Y){
	if(X.rows() != Y.rows()){
		throw invalid_argument("to compute a Gram Matrix both input matrices must have the same number of rows.");
	}
	int M = X.cols();
	int N = Y.cols();
	MatrixXd result = MatrixXd(M,N);
	for(int i = 0; i < M; ++i){
		for(int j = 0; j < N; ++j){
			result(i,j) = this->k(X.col(i),Y.col(j));
		}
	}
	return result;
}

vector< vector<double> > kernel::py_gram_matrix(vector< vector<double> > X, vector< vector<double> > Y){
	MatrixXd ans = gram_matrix(stl_to_eigen(X), stl_to_eigen(Y));
	return eigen_to_stl(ans);
}

/* Implementation for the linear_kernel class */

linear_kernel::linear_kernel(double c):
	_c(c)
	{}

// http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear
double linear_kernel::k(VectorXd x_i, VectorXd y_j){
	return x_i.transpose() * y_j + _c;
}

/* Implementation for the polynomial_kernel class */

polynomial_kernel::polynomial_kernel(double a, double c, double d):
	_a(a),
	_c(c),
	_d(d)
	{}

//http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#polynomial
double polynomial_kernel::k(VectorXd x_i, VectorXd y_j){
	double dot = x_i.transpose() * y_j;
	double base = _a * dot + _c;
	return pow(base,_d);
}

/* Implementation for the gaussian_kernel class */

gaussian_kernel::gaussian_kernel(double s):
	_s(s)
	{}

// http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#gaussian
double gaussian_kernel::k(VectorXd x_i, VectorXd y_j){
	return exp(-1 * (x_i - y_j).array().square().sum() / (2 * _s * _s));
}