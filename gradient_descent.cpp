/*Impementation of gradient_descent*/

#include "gradient_descent.h"

/**************************************************************
 * Implementation for the base class optomization_solver_base *
 **************************************************************/

optomization_solver_base::optomization_solver_base(){}

vector<double> optomization_solver_base::get_loss(){
	return loss_values;
}

vector<double> optomization_solver_base::eigen_to_stl(VectorXd v){
	vector<double> t(v.rows());
	for(int i = 0; i < v.rows(); ++i){
		t[i] = v(i);
	}
	return t;
}

VectorXd optomization_solver_base::stl_to_eigen(vector<double> v){
	VectorXd t(v.size());
	for(int i = 0; i < v.size(); ++i){
		t(i) = v[i];
	}
	return t;
}

// this method assumes the v has dimensions ixj, however because v is a vector of vectors
// it is possible that the vectors in v have different sizes. 
// ie:
// [[1,2,3],
//  [1,2],
//  [1,2,3]]
// therfore we're going to check that this property holds within the inner loop 
// and throw an exception if necessary
MatrixXd optomization_solver_base::stl_to_eigen(vector< vector<double> > v){
	MatrixXd m(v.size(),v[0].size());
	for(int i = 0; i < v.size(); ++i){
		// check the precondition
		if(v[0].size() != v[i].size()){
			throw invalid_argument("cannot convert input values to Eigen MatrixXd. All rows must have the same number of values");
		}

		for(int j = 0; j < v[i].size(); ++j){
			m(i,j) = v[i][j];
		}
	}
	return m;
}

/*******************************************************
 * Implementation for the batch_gradient_descent class *
 *******************************************************/

// determines if the values in w are beneath the precision value
bool batch_gradient_descent::done(VectorXd w, double precision){
	// convert w to an Array so we can do coefficent-wise ops like abs()
	ArrayXd temp = w.array();
	temp = temp.abs();
	return (temp < precision).all();
}

// default constructor to appease Cython
batch_gradient_descent::batch_gradient_descent(){}

// construction using arbitrary size data
batch_gradient_descent::batch_gradient_descent(MatrixXd X, VectorXd y, model* M):
	_X(X),
	_y(y)
	{m = M;}
	
batch_gradient_descent::batch_gradient_descent(vector< vector<double> > X, vector<double> y, model* M):
	batch_gradient_descent(stl_to_eigen(X),stl_to_eigen(y), M)
	{}
// does the actual fitting using gradient descent
// params: 
VectorXd batch_gradient_descent::fit(VectorXd init, double gamma, double precision){
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
		w_k1 = w_k;
		w_k -= gamma*m->gradient(w_k1, _X, _y);
		diff = w_k1 - w_k;
		loss = m->loss(w_k, _X, _y);
		if(loss == numeric_limits<double>::infinity()){
			throw runtime_error("we have diverged!");
		}
		loss_values.push_back(loss);
	}
	return w_k;
}

// a slightly different version of fit so that I don't have 
// to wrap Eigen for Cython
// It calls the normal fit function and moves it into a STL vector so Cython can convert it to numpy
vector<double> batch_gradient_descent::py_fit(vector<double> init, double gamma, double precision){
	VectorXd ans = fit(stl_to_eigen(init), gamma, precision);
	return eigen_to_stl(ans);
}

/************************************************************
 * Implementation for the stochastic_gradient_descent class *
 ************************************************************/

stochastic_gradient_descent::stochastic_gradient_descent(){}

// TODO: implement, issue #7 on github
VectorXd stochastic_gradient_descent::fit(VectorXd prev, double gamma){
	// compute the gradient but with only the given data
	VectorXd result;
	return result;
}

vector<double> stochastic_gradient_descent::py_fit(vector<double> prev, double gamma){
	VectorXd ans = fit(stl_to_eigen(prev), gamma);
	return eigen_to_stl(ans);
}