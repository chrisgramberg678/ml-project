#include <stdlib.h>
#include <time.h>
#include "gradient_descent.h"

VectorXd test_prep(MatrixXd& X, VectorXd& y, VectorXd& init, int coefficients, int points){
	// create the specified number of coefficients and 
	//stick them in a vector so we know the answer
	VectorXd coefs(coefficients);
	for(int i = 0; i < coefficients; ++i){
		coefs(i) = (rand()%200 - 100)/10.0; // we want coeffecients in the range[-10,10]
	}
	// set X to be coefficients by data points
	X.resize(coefficients,points);
	// fill X with some random values between 1 and 10
	for(int i = 0; i < coefficients; ++i){
		for(int j = 0; j < points; ++j){
			X(i,j) = rand() % 10;
		}
	}
	// fill y with some values by multiplying X and the coefficients
	// and adding some noise
	y = coefs.transpose() * X;
	// y.resize(points);
	for(int i = 0; i < points; ++i){
		y(i) += (rand() % 200 - 100) / 100.0;
		// y(i) = rand() % 2;
	}
	// set init to have the correct dimensions and set it to all 0's
	init.resize(coefficients);
	for(int i = 0; i < coefficients; ++i){
		init(i) = 0;
	}
	return coefs;
}


// pick a w randomly and generate based on that
// i should be able to find some bernoulli rng to set the probability of one being what we set earlier

int main(){
	srand(1);
	double gamma = .001;
	double precision = .0000001;
	// counts the number of times we're off by error
	int bad = 0;
	double error = .2;
	linear_least_squares_model m;
	// binary_logistic_regression_model m;
	model* M = &m;

	for(int i = 1; i < 20; ++i){
		cout << "coefficients: " << i << endl;
		MatrixXd x;
		VectorXd y, ans, init;
		ans = test_prep(x, y, init, i, i*20);
		batch_gradient_descent gd(x,y,M);
		VectorXd res;
		try {
			res = gd.fit(init, gamma, "step_precision", precision);
			ArrayXd temp = (res - ans).array();
			// if our calculated coefs are within an error of the answer we're good
			if((temp.abs() < error).all()){
				cout << "GOOD" << endl;
			}
			else{
				++bad;
				cout << "OFF" << endl;
				cout << "result " << endl << res << endl;
				cout << "ans " << endl << ans << endl;
			}
		}
		catch(std::exception& e){
			cout << "exception: " << e.what() << endl;
		}
	}
	cout << bad << " were off by more than " << error << endl;
}