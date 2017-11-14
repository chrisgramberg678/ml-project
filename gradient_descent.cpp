/*Impementation of gradient_descent*/

#include "gradient_descent.h"

/**************************************************************
 * Implementation for the base class optomization_solver_base *
 **************************************************************/

optomization_solver_base::optomization_solver_base(){}

vector<double> optomization_solver_base::get_loss_values(){
	return loss_values;
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

batch_gradient_descent::batch_gradient_descent(){}

batch_gradient_descent::batch_gradient_descent(Map<MatrixXd> X, Map<VectorXd> y, model* M):
	_X(X),
	_y(y)
	{m = M;}	
/* does the actual fitting using gradient descent
 * params: init - the starting point for the optomization
 * 		   gamma - the step size
 *         convergence_type - either "iterations", "loss_precision", "step_precision", or "none"
 *         conv - either the max iterations or the precision provided by the caller
 */
VectorXd batch_gradient_descent::fit(Map<VectorXd> init, double gamma, string convergence_type, double conv){
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

/************************************************************
 * Implementation for the stochastic_gradient_descent class *
 ************************************************************/

stochastic_gradient_descent::stochastic_gradient_descent(){}

stochastic_gradient_descent::stochastic_gradient_descent(model* M){m = M;}

// take some data X and labels y and return the value of the next gradient step
VectorXd stochastic_gradient_descent::fit(Map<VectorXd> prev, double gamma, Map<MatrixXd> X, Map<VectorXd> y){
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
