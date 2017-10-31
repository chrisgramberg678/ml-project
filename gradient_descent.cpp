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

// determine if we're done based on the convergence type
bool batch_gradient_descent::done(string type, double conv, int iteration, VectorXd step_diff, double loss_diff){
	if(type.compare("step_precision") == 0){
		// convert step_diff to an Array so we can do coefficent-wise ops like abs()
		ArrayXd temp = step_diff.array();
		temp = temp.abs();
		return (temp < conv).all();
	}
	else if(type.compare("loss_precision") == 0){
		return abs(loss_diff) < conv;
	}
	else if(type.compare("iterations") == 0){
		return  iteration > conv;
	}
	else{
		throw invalid_argument("invalid convergence type: " + type);
	}
}

// variant of done which checks if the change in loss is less than a precision value
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
	
/* does the actual fitting using gradient descent
 * params: init - the starting point for the optomization
 * 		   gamma - the step size
 *         convergence_type - either "iterations", "loss_precision", "step_precision", or "none"
 *         conv - either the max iterations or the precision provided by the caller
 */
VectorXd batch_gradient_descent::fit(VectorXd init, double gamma, string convergence_type, double conv){
	// if(init.rows() != _X.rows()){
	// 	throw invalid_argument("initial values must have the same size as the number of coefficients");
	// }
	if(convergence_type.compare("none") != 0 && convergence_type.compare("loss_precision") != 0 && convergence_type.compare("step_precision") != 0 && convergence_type.compare("iterations") != 0){
		throw invalid_argument("invalid convergence type: " + convergence_type);
	}
	// if there is no convergence type provided then we'll just do 1 million iterations
	if(convergence_type.compare("none") == 0){
		convergence_type = "iterations";
		conv = 1000000;
	}
	// these are used for iteration
	VectorXd prev, next, step_diff(init.rows());
	// starting values for w the weights we're solving for
	next = init;
	// fill step_diff with ones so we can get false from done the first time
	for(int i = 0; i < next.rows(); ++i){
		step_diff(i) = 1;
	}
	double loss = 10000000;
	double loss_diff = 10000000;
	loss_values.push_back(loss);
	int i = 0;
	// we'll go until we stop from the convergence condition or we hit a billion iterations, whichever is first
	while( !done(convergence_type, conv, i, step_diff, loss_diff) && i < 1000000000){
		// uncomment to diagnose infinite looping
		// if(i % 10 == 0){
		// 	cout << "i: " << i << endl;
		// 	cout << "loss: " << loss << endl;
		// }
		++i;
		prev = next;
		next = prev - gamma*m->gradient(prev, _X, _y);
		loss = m->loss(next, _X, _y);
		if(loss == numeric_limits<double>::infinity()){
			throw runtime_error("we have diverged!");
		}
		loss_values.push_back(loss);
		// update the diffs for the convergence check
		step_diff = prev - next;
    	loss_diff = loss_values[loss_values.size() - 1] - loss_values[loss_values.size() - 2];
    }
	return next;
}

// a slightly different version of fit so that I don't have 
// to wrap Eigen for Cython
// It calls the normal fit function and moves it into a STL vector so Cython can convert it to numpy
vector<double> batch_gradient_descent::py_fit(vector<double> init, double gamma, string convergence_type, double conv){
	VectorXd ans = fit(stl_to_eigen(init), gamma, convergence_type, conv);
	return eigen_to_stl(ans);
}

/************************************************************
 * Implementation for the stochastic_gradient_descent class *
 ************************************************************/

stochastic_gradient_descent::stochastic_gradient_descent(){}

stochastic_gradient_descent::stochastic_gradient_descent(model* M){m = M;}

// take some data X and labels y and return the value of the next gradient step
VectorXd stochastic_gradient_descent::fit(VectorXd prev, double gamma, MatrixXd X, VectorXd y){
	if(m->parametric()){
		// compute the gradient based on the given data
		VectorXd result = prev - gamma * m->gradient(prev, X, y);
		// calculate the loss and add it to loss_values
		double loss = m->loss(result, X, y);
		loss_values.push_back(loss);
		return result;
	}
	else{
		// first compute the functional gradient given X and y
		VectorXd f_gradient = m->gradient(prev, X, y);
		VectorXd result = VectorXd::Zero(f_gradient.rows());
		for(int i = 0; i < result.size(); ++i){
			// since our result has one more weight than prev the last value will just 
			// be multiplied by -gamma
			if(i < prev.size()){
				result(i) = prev(i) - gamma * f_gradient(i);
			}
			else{
				result(i) = - gamma * f_gradient(i);
			}
		}
		// calculate the loss and add it to loss_values
		double loss = m->loss(result, X, y);
		loss_values.push_back(loss);
		return result;
	}		
}

vector<double> stochastic_gradient_descent::py_fit(vector<double> prev, double gamma, vector< vector<double> > X, vector<double> y){
	VectorXd ans = fit(stl_to_eigen(prev), gamma, stl_to_eigen(X), stl_to_eigen(y));
	return eigen_to_stl(ans);
}

