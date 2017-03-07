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
}