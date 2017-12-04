// abstract class that outlines the functionality of a model.
// for example, we might use the linear least sqaures model for linear regression 
// or a binary logistic model to do binary clasification

#include "kernel.h"

class model{ 
	private:
		bool _parametric;
	protected:
		VectorXd _weights;
	public:
		// used to determine how SGD calculates the next value
		model();
		model(bool parametric);
		bool parametric();
		void init_weights(int num_weights);
		void init_weights(VectorXd init);
		virtual VectorXd gradient(MatrixXd X, VectorXd y);
		virtual double loss(MatrixXd X, VectorXd y);
		// TODO: have this return a tuple of VectorXds (label, probability)
		// currently it only returns the label
		virtual VectorXd predict(Map<MatrixXd> X);
		friend class batch_gradient_descent;
		friend class stochastic_gradient_descent;
};

class linear_least_squares_model : public model{
	public:
		linear_least_squares_model();
		VectorXd gradient(MatrixXd  X, VectorXd y);
		double loss(MatrixXd X, VectorXd y);
		VectorXd predict(Map<MatrixXd> X);
};

class binary_logistic_regression_model : public model{
	public:
		binary_logistic_regression_model();
		VectorXd gradient(MatrixXd  X, VectorXd y);
		double loss(MatrixXd X, VectorXd y);
		VectorXd predict(Map<MatrixXd> X);
};

class kernel_binary_logistic_regression_model: public model{
	protected:
		double _lambda;
		kernel* _k;
		MatrixXd _KXX;
		bool first;
		// the training set is needed to make predictions in this model
		MatrixXd _X_train;
	public:
		kernel_binary_logistic_regression_model();
		kernel_binary_logistic_regression_model(kernel* k, double lambda);
		VectorXd gradient(MatrixXd  X, VectorXd y);
		double loss(MatrixXd X, VectorXd y);
		VectorXd predict(Map<MatrixXd> X);
};

class stochastic_kernel_logistic_regression_model: public kernel_binary_logistic_regression_model{
	private:
		MatrixXd _dictionary;
		double f(VectorXd X);
		// add weights
	public:
		stochastic_kernel_logistic_regression_model();
		stochastic_kernel_logistic_regression_model(kernel* k, double lambda);
		VectorXd gradient(MatrixXd X, VectorXd y);
		double loss(MatrixXd X, VectorXd y);	
		VectorXd predict(Map<MatrixXd> X);
};