/*First impementation of gradient descent*/

#include <iostream>
#include <Eigen/Dense>
#include <math.h>
#include <vector>
using namespace std;
using Eigen::Vector2d;

vector<int> x = {1,2,3,4};
vector<int> y = {6,5,7,10};

// computes a new vector ab based on the summation of the derivatives with 
// x and y subbed in from the training data
Vector2d dL(Vector2d ab){
	double da = 15*ab(0) + 5*ab(1) - 38.5;
	double db = 2*ab(1) + 5*ab(0) - 14;
	Vector2d temp;
	temp << da,db;
	// cout << "tempDL: " << endl << temp << endl;
	return temp;
}

// dL/da = 1/N sum(-2x(y-xa-b))
// where x and y come from the training data and N is the size of x and y
double dLda(double a, double b){
	double sum = 0;
	for(int i = 0; i < x.size(); ++i){
		sum += (-2)*(x[i])*(y[i]-((a*x[i])+b));
	}
	sum /= x.size();
	return sum;
}

// dL/db = 1/N sum(-2(y-ax-b))
// where x and y come from the training data and N is the size of x and y
double dLdb(double a, double b){
	double sum = 0;
	for(int i = 0; i < x.size(); ++i){
		sum += (-2)*(y[i]-((a*x[i])+b));
	}
	sum /= x.size();
	return sum;
}

// computes a new vector by doing the summation over the training data 
Vector2d dLs(Vector2d ab){
	double a = dLda(ab(0),ab(1));
	double b = dLdb(ab(0),ab(1));
	Vector2d temp;
	temp << a,b;
	return temp;
}

// L(a,b) = 1/N sum (y-ax-b)^2
// we know this is working when the value of the loss function gets smaller
double Loss(double a, double b){
	double sum = 0;
	for(int i = 0; i < x.size(); ++i){
		double temp = y[i] - ((a*x[i]) + b);
		sum += (temp*temp);
	}
	sum /= x.size();
	return sum;
}

int main(){
// step size
double gamma = .001;
// when |x_old - x_new| > precision we're there
double precision = .000000001;
Vector2d x_old, x_new, diff;
// these don't matter b/c they are trashed once we get into the while
x_old << 10,10;
// starting a and b will be (0,0)
x_new << 0,0;
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
}