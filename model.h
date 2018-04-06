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
		VectorXd get_weights();
		virtual void update_weights(VectorXd new_w);
		// for the Cython interface
		virtual double loss(Map<MatrixXd> X, Map<VectorXd> y);
		virtual void update_weights(VectorXd new_w, MatrixXd X);
		virtual double loss(MatrixXd X, VectorXd y) = 0;
		virtual VectorXd gradient(MatrixXd X, VectorXd y) = 0;
		// TODO: consider having this return a tuple of VectorXds (label, probability)
		// currently it only returns the label
		virtual VectorXd predict(Map<MatrixXd> X) = 0;
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
		MatrixXd _KDD;
		MatrixXd _KDD_inverse;
		double _err_max;
		double f(VectorXd X);
		void update_dictionary(MatrixXd X);
		void update_KDD();
		void update_KDD_inverse();
		void prune_dictionary();
		MatrixXd remove_col_from_dict(MatrixXd d, int i);
		MatrixXd remove_sample_from_Kdd(MatrixXd Kdd, int i);
		MatrixXd remove_sample_from_inverse(MatrixXd old_inverse, int i);
		
	public:
		stochastic_kernel_logistic_regression_model();
		stochastic_kernel_logistic_regression_model(kernel* k, double lambda, double err_max);
		VectorXd gradient(MatrixXd X, VectorXd y);
		double loss(MatrixXd X, VectorXd y);	
		VectorXd predict(Map<MatrixXd> X);
		void update_weights(VectorXd new_w, MatrixXd X);
		MatrixXd dictionary();
};