// abstract class that outlines the functionality of a model.
// for example, we might use the linear least sqaures model for linear regression 
// or a binary logistic model to do binary clasification

#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Model{
	public:
		virtual VectorXd gradient(VectorXd w, MatrixXd X, VectorXd y) = 0;
		virtual double loss(VectorXd w, MatrixXd X, VectorXd y) = 0;
};

class linear_least_squares_model : public Model{
	VectorXd gradient(VectorXd w, MatrixXd  X, VectorXd y);
	double loss(VectorXd w, MatrixXd X, VectorXd y);
};

class binary_logistic_regression_model : public Model{
	VectorXd gradient(VectorXd w, MatrixXd  X, VectorXd y);
	double loss(VectorXd w, MatrixXd X, VectorXd y);
};