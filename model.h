// abstract class that outlines the functionality of a model.
// for example, we might use the linear least sqaures model for linear regression 
// or a binary logistic model to do binary clasification

#include <Eigen/Dense>
#include <iostream>
#include "math.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class model{
	public:
		model();
		virtual VectorXd gradient(VectorXd w, MatrixXd X, VectorXd y);
		virtual double loss(VectorXd w, MatrixXd X, VectorXd y);
};

class linear_least_squares_model : public model{
	public:
		linear_least_squares_model();
		VectorXd gradient(VectorXd w, MatrixXd  X, VectorXd y);
		double loss(VectorXd w, MatrixXd X, VectorXd y);
};

class binary_logistic_regression_model : public model{
	public:
		binary_logistic_regression_model();
		VectorXd gradient(VectorXd w, MatrixXd  X, VectorXd y);
		double loss(VectorXd w, MatrixXd X, VectorXd y);
};
