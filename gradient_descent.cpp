/*Impementation of gradient_descent*/

#include "gradient_descent.h"

// determines if the values in w are beneath the precision value
bool gradient_descent::done(VectorXd w, double precision){
	// convert w to an Array so we can do coefficent-wise ops like abs()
	ArrayXd temp = w.array();
	temp = temp.abs();
	return (temp < precision).all();
}

// default constructor to appease Cython
gradient_descent::gradient_descent(){}

// construction using arbitrary size data
gradient_descent::gradient_descent(MatrixXd X, VectorXd y):
	_X(X),
	_y(y)
	{}
	
gradient_descent::gradient_descent(vector< vector<double> > X, vector<double> y):
	gradient_descent(stl_to_eigen(X),stl_to_eigen(y))
	{}
// does the actual fitting using gradient descent
// params: 
VectorXd gradient_descent::fit(VectorXd init, double gamma, double precision, model* M, bool verbose){
	if(init.rows() != _X.rows()){
		throw invalid_argument("initial values must have the same size as the number of coefficients");
	}
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
   			// cout << "iteration: " << i++ << endl;
			cout << "Loss: " << loss << endl;
			// cout << "gradient: " << w_k << endl;
		}
		w_k1 = w_k;
		w_k -= gamma*M->gradient(w_k1, _X, _y);
		diff = w_k1 - w_k;
		loss = M->loss(w_k, _X, _y);
		if(loss == numeric_limits<double>::infinity()){
			throw runtime_error("we have diverged!");
		}
	}
	if(verbose){
		cout << "coefficients: " << endl << w_k << endl;
	}
	return w_k;
}

vector<double> gradient_descent::eigen_to_stl(VectorXd v){
	vector<double> t(v.rows());
	for(int i = 0; i < v.rows(); ++i){
		t[i] = v(i);
	}
	return t;
}

VectorXd gradient_descent::stl_to_eigen(vector<double> v){
	VectorXd t(v.size());
	for(int i = 0; i < v.size(); ++i){
		t(i) = v[i];
	}
	return t;
}

// this method assumes the v has dimensions ixj, however because v is a vector of vectors
// it is possible that the vectors in v have different sizes. 
MatrixXd gradient_descent::stl_to_eigen(vector< vector<double> > v){
	MatrixXd m(v.size(),v[0].size());
	for(int i = 0; i < v.size(); ++i){
		for(int j = 0; j < v[i].size(); ++j){
			m(i,j) = v[i][j];
		}
	}
	return m;
}

// a slightly different version of fit so that I don't have 
// to wrap Eigen for Cython
// It calls the normal fit function and moves it into a STL vector so Cython can convert it to numpy
vector<double> gradient_descent::py_fit(vector<double> init, double gamma, double precision, model* M){
	VectorXd ans = fit(stl_to_eigen(init), gamma, precision, M);
	return eigen_to_stl(ans);
}