/*Impementation of gradient_descent*/

#include "gradient_descent.h"

// dL/da = 1/N sum(-2x(y-xa-b))
// where x and y come from the training data and N is the size of x and y

double gradient_descent::dLda(double a, double b){
	double sum = 0;
	for(int i = 0; i < x.size(); ++i){
		sum += (-2)*(x[i])*(y[i]-((a*x[i])+b));
	}
	sum /= x.size();
	return sum;	
}

// dL/db = 1/N sum(-2(y-ax-b))
// where x and y come from the training data and N is the size of x and y
double gradient_descent::dLdb(double a, double b){
	double sum = 0;
	for(int i = 0; i < x.size(); ++i){
		sum += (-2)*(y[i]-((a*x[i])+b));
	}
	sum /= x.size();
	return sum;
}

// computes a new vector by doing the summation over the training data 
Vector2d gradient_descent::dL(Vector2d ab){
	double a = dLda(ab(0),ab(1));
	double b = dLdb(ab(0),ab(1));
	Vector2d temp;
	temp << a,b;
	return temp;
}

// L(a,b) = 1/N sum (y-ax-b)^2
// we know this is working when the value of the loss function gets smaller
double gradient_descent::Loss(double a, double b){
	double sum = 0;
	for(int i = 0; i < x.size(); ++i){
		double temp = y[i] - ((a*x[i]) + b);
		sum += (temp*temp);
	}
	sum /= x.size();
	return sum;
}

gradient_descent::gradient_descent(vector<int> n, vector<int> m):
	x(n),
	y(m)
	{}
	
// does the actual fitting using gradient descent
// params: 
Vector2d gradient_descent::fit(Vector2d ab, double gamma, double precision, bool verbose){
	Vector2d x_old, x_new, diff;
	// these don't matter b/c they are trashed once we get into the while
	x_old << ab(0)+1, ab(1)+1;
	// starting a and b will be (0,0)
	x_new = ab;
	diff = x_old - x_new;
	double loss = 10000000;
	int i = 0;
	while((fabs(diff(0)) > precision || fabs(diff(1)) > precision)){
   		cout << "iteration: " << i++ << endl;
		cout << "Loss " << loss << endl;
		x_old = x_new;
		x_new += -gamma*dL(x_old);
		diff = x_old - x_new;
		loss = Loss(x_new(0),x_new(1));
	}
	cout << "coefficients: " << endl << x_new << endl;
	return x_new;
}