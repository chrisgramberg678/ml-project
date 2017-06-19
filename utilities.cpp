#include "utilities.h"

vector<double> eigen_to_stl(VectorXd v){
	vector<double> t(v.rows());
	for(int i = 0; i < v.rows(); ++i){
		t[i] = v(i);
	}
	return t;
}

vector< vector<double> > eigen_to_stl(MatrixXd m){
	vector< vector<double> > v(m.rows());
	// initialize all of the vectors
	for(int i = 0; i < m.rows(); ++i){
		vector<double> temp(m.cols());
		v[i] = temp;
	}
	for(int i = 0; i < m.rows(); ++i){
		for(int j = 0; j < m.cols(); ++j){
			v[i][j] = m(i,j);
		}
	}
	return v;
}

VectorXd stl_to_eigen(vector<double> v){
	VectorXd t(v.size());
	for(int i = 0; i < v.size(); ++i){
		t(i) = v[i];
	}
	return t;
}

// this method assumes the v has dimensions ixj, however because v is a vector of vectors
// it is possible that the vectors in v have different sizes. 
// ie:
// [[1,2,3],
//  [1,2],
//  [1,2,3]]
// therfore we're going to check that this property holds within the inner loop 
// and throw an exception if necessary
MatrixXd stl_to_eigen(vector< vector<double> > v){
	MatrixXd m(v.size(),v[0].size());
	for(int i = 0; i < v.size(); ++i){
		// check the precondition
		if(v[0].size() != v[i].size()){
			throw invalid_argument("cannot convert input values to Eigen MatrixXd. All rows must have the same number of values");
		}

		for(int j = 0; j < v[i].size(); ++j){
			m(i,j) = v[i][j];
		}
	}
	return m;
}