/* implementation of various kernels and the gram_matrix method in the parnet kernel class  */

#include "kernel.h"

/*
 Implementation for the gram_matrix method that all children of kernel use
 It is dependent on the virtual method k() which computes the kernel between two vectors 
*/

MatrixXd kernel::gram_matrix(MatrixXd X, MatrixXd Y){
	return MatrixXd();
}

/* Implementation for the linear_kernel class */

linear_kernel::linear_kernel(double c)
	_c(c)
	{}

double linear_kernel::k(VectorXd x_i, VectorXd y_j){
	return x_i.transpose() * y_j + _c;
}

/* Implementation for the polynomial_kernel class */

polynomial_kernel::polynomial_kernel(double a, double c, double d)
	_a(a),
	_c(c),
	_d(d)
	{}

double polynomial_kernel::k(VectorXd x_i, VectorXd y_j){
	return pow((_a * (x_i.transpose() * y_j) +_c),_d);
}

/* Implementation for the gaussian_kernel class */

gaussian_kernel::gaussian_kernel(double s)
	_s(s)
	{}

double gaussian_kernel::k(VectorXd x_i, VectorXd y_j){
	return exp((x_i -y_j).array().square().sum() / 2 * _s * _s);
}