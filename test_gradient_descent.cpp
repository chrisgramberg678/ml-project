#include "gradient_descent.h"

int main(){
	double gamma = .001;
	double precision = .000000001;
	vector<int> x = {1,2,3,4};
	vector<int> y = {6,5,7,10};
	Vector2d ab;
	ab << 0,0;
	gradient_descent gd(x,y);
	Vector2d res = gd.fit(ab,gamma,precision,true);
	cout << "result " << endl << res << endl;
	vector<int> ab2 = {0,0};
	vector<double> r = gd.py_fit(ab2,gamma,precision,true);
	cout << "result of cython friendly version " << endl << r[0] << endl << r[1] << endl;
}