/* Simple sanity check for the kernel implementations */

#include "kernel.h"

int main(){
	linear_kernel lk = linear_kernel(0);
	// these params should make this act exactly like the linear kernel
	polynomial_kernel pk = polynomial_kernel(1,0,1);
	// not totally sure what this will do
	gaussian_kernel gk = gaussian_kernel(.5);
	cout << "construction worked" << endl;
	MatrixXd X(2,2);
	MatrixXd Y(2,1);
	X(0,0) = 1;
	X(0,1) = 2;
	X(1,0) = 3;
	X(1,1) = 4;
	cout << "X:\n" << X << endl;
	Y(0,0) = 5;
	Y(1,0) = 6;
	cout << "Y:\n" << Y << endl;
	cout << "K(X,Y) for a linear kernel with c = 0:\n" << lk.gram_matrix(X,Y) << endl;
	cout << "K(X,Y) for a polynomial kernel with a = 1, c = 0, d = 1:\n" << pk.gram_matrix(X,Y) << endl;
	cout << "K(X,Y) for a gaussian kernel with s = .5:\n" << gk.gram_matrix(X,Y) << endl;
}