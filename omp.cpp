/*
	Test file for getting omp working before adding it into any models
	run with: g++ -std=c++11 -I /usr/local/include/eigen3 kernel.cpp omp.cpp -o omp && ./omp
 */
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <random>
#include <vector>
#include <stdexcept>
#include "kernel.h"
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

MatrixXd add_cols_to_inverse(MatrixXd _KDD, MatrixXd _KDD_inverse, MatrixXd _dictionary, kernel* _k){
	// we'll update the iverse matrix iteratively for each new dictionary value 
	while(_KDD_inverse.cols() < _KDD.cols()){
		int c = _KDD_inverse.cols();
		// the one column version of the inverse is a 1x1 matrix with the value (K_DD(0,0))^-1. because K(x,x) = 1, (K_DD(0,0))^-1 = 1
		if(c == 0){
			_KDD_inverse = MatrixXd(1,1);
			_KDD_inverse(0,0) = 1.0/_KDD(0,0);
		}
		// add a single row and column
		else{
			VectorXd v = _dictionary.col(c);
			// create some intermediary values
			VectorXd u1 = _k->gram_matrix(_dictionary.leftCols(c), v);
			VectorXd u2 = _KDD_inverse * u1;
			double d = 1;///(v.transpose() * v - u1.transpose() * u2).value();
			VectorXd u3 = d * u2;
			MatrixXd top_left = _KDD_inverse + d * u2 * u2.transpose();
			// create _KDD_inverse from the intermediary values
			MatrixXd new_KDD_inverse = MatrixXd(c + 1, c + 1);
			new_KDD_inverse.topLeftCorner(c, c) = top_left;
			new_KDD_inverse.topRightCorner(c, 1) = -1 * u3;
			new_KDD_inverse.bottomLeftCorner(1, c) = -1 * u3.transpose();
			new_KDD_inverse(c, c) = d;
			_KDD_inverse = new_KDD_inverse;
		}
	}
	return _KDD_inverse;
}

MatrixXd remove_col_from_inverse(MatrixXd old_inverse, int i){
	// permute the ith column and ith column to the last column and last row
	VectorXd ith_row = old_inverse.row(i);
	VectorXd ith_col = old_inverse.col(i);
	for(int j = 0; j < old_inverse.rows(); ++j){
		if(j > i){
			old_inverse.row(j - 1) = old_inverse.row(j);
			old_inverse.col(j - 1) = old_inverse.col(j);
		}
	}
	old_inverse.bottomRows(1) = ith_row.transpose();
	old_inverse.rightCols(1) = ith_col;
	MatrixXd top_left = old_inverse.topLeftCorner(old_inverse.rows() - 1, old_inverse.cols() - 1);
	double d = old_inverse(old_inverse.rows() - 1, old_inverse.rows() - 1);
	VectorXd u3 = -1 * old_inverse.topRightCorner(old_inverse.rows() - 1, 1);
	VectorXd u2 = (1/d) * u3;
	MatrixXd new_inverse = top_left - (d * (u2 * u2.transpose()));
	return new_inverse;
}

MatrixXd remove_col_from_dict(MatrixXd d, int i){
	if(i > d.cols() - 1 || i < 0){
		stringstream ss;
		ss << "Cannot remove column " << i << " from dictionary of length " << d.cols() << ".";
		throw std::invalid_argument(ss.str());
	}
	// permute the target column to the end
	for(int j = 0; j < d.cols(); ++j){
		if(j > i){
			d.col(j - 1) = d.col(j);
		}
	}
	// return all but the last column
	return d.leftCols(d.cols() - 1);
}

void compare_inverses(){
	MatrixXd d_t1(2,5);
	d_t1 << .1, .3, .5, .7, .3,
			.2, .4, .6, .8, .4;
	
	MatrixXd d_t(2,4);
	d_t << .1, .3, .5, .7,
		   .2, .4, .6, .8;
	
	gaussian_kernel gk(.5);
	MatrixXd KDD_t = gk.gram_matrix(d_t, d_t);
	MatrixXd KDD_t1 = gk.gram_matrix(d_t1, d_t1);
	MatrixXd constructive_inverse_t, constructive_inverse_t1;
	constructive_inverse_t = add_cols_to_inverse(KDD_t, constructive_inverse_t, d_t, &gk);
	constructive_inverse_t1 = add_cols_to_inverse(KDD_t1, constructive_inverse_t1, d_t1, &gk);
	MatrixXd destructive_inverse_t = remove_col_from_inverse(constructive_inverse_t1, 4);
	cout << "inverse built constructively:\n" << constructive_inverse_t << endl;
	cout << "inverse built destructivley:\n" << destructive_inverse_t << endl;
	cout << "*********************************\n";
	// compare the inverse * the kernel matrx to the identity using the matrix norm
}

// tests the inversion of a matrix built by add_cols_to_inverse() using a dictionary of size rows, cols
// here, rows are features and columns are samples, this means that our Kernel matrix and the corresponding
// inverse will be cols x cols
// returns ((Kdd_inverse * Kdd) - MatrixXd::Identity()).norm() < threshold 
bool invert_Kdd_test(int rows, int cols, double threshold, std::vector<double> &norms, bool verbose=false){
	gaussian_kernel gk(.5);

	std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd{0,5};

	MatrixXd d(rows, cols);

	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j)
			d(i,j) = nd(gen);
	}
	MatrixXd Kdd = gk.gram_matrix(d,d);
	MatrixXd inverse_Kdd = add_cols_to_inverse(Kdd, MatrixXd(), d, &gk);
	MatrixXd approx_identity = Kdd * inverse_Kdd;
	MatrixXd identity = MatrixXd::Identity(Kdd.rows(),Kdd.cols());
	// the norm of the difference between the approx_identity and the the identity should be small
	MatrixXd diff = approx_identity - identity;
	double norm = diff.norm();
	norms.push_back(norm);
	if(verbose){
		std::stringstream info;
		info << "Dictonary - Rows: " << rows << ". Cols: " << cols << ".\n";
		info << d << endl << endl;
		info << "Kdd:\n" << Kdd << endl << endl;
		info << "Inverse:\n" << inverse_Kdd << endl << endl;
		info << "Eigen inverse:\n" << Kdd.inverse() << endl << endl;
		info << "Kdd * inverse_Kdd:\n" << approx_identity << endl << endl;
		info << "Difference from identity:\n" << diff << endl << endl;
		info << "Norm < threshold: " << norm << " < " << threshold << " -> " << ((norm < threshold) ? "True" : "False") << endl; 
		cout << info.str() << endl;
	}
	return norm < threshold;
}


bool remove_col_from_dict_test(){
	MatrixXd d(2,5);
	d << 1.6, 2.8, .5, -.7, 2.8,
		-3.2, .2, -.5, 1.7, .2;

	// remove a column in the middle
	MatrixXd removed_middle(2,4);
	removed_middle << 1.6, 2.8, -.7, 2.8,
					 -3.2, .2, 1.7, .2;
	if(!removed_middle.isApprox(remove_col_from_dict(d, 2))){
		return false;
	}
	
	// remove the first column
	MatrixXd removed_first(2,4);
	removed_first << 2.8, .5, -.7, 2.8,
					.2, -.5, 1.7, .2;
	if(!removed_first.isApprox(remove_col_from_dict(d,0))){
		return false;
	}
	
	// remove the last column
	MatrixXd removed_last(2,4);
	removed_last << 1.6, 2.8, .5, -.7,
					-3.2, .2, -.5, 1.7;
	if(!removed_last.isApprox(remove_col_from_dict(d,4))){
		return false;
	}

	// intentionally pass an invalid index
	try{
		remove_col_from_dict(d,-1);
		return false;
	}
	catch (const invalid_argument& ia){
		// everything is fine
	}
	try{
		remove_col_from_dict(d,5);
		return false;
	}
	catch (const invalid_argument& ia){
		// everything is fine
	}	
	return true;
}

// tests remove_col_from_inverse() by removing each column from dictionary and checking that the resulting inverse is valid
// this is done by checking ((Kdd_inverse * Kdd) - MatrixXd::Identity()).norm() < threshold for each kernel matrix and associated inverse computed 
// by excluding a single value from the dictionary
bool remove_col_test(){

}


int main(){

	cout << "remove_col_from_dict works: " << (remove_col_from_dict_test()?"True":"False") << endl;

	int samples = 10;
	for(int s = 2; s < samples; ++s){
		int tries = 200;
		std::vector<double> norms;
		for(int t = 0; t < tries; ++t){
			invert_Kdd_test(2, s, 1, norms);
		}
		double mean = 0;
		for(auto n : norms){
			mean += n;
		}
		mean/=tries;
		cout << "Samples: " << s << endl;
		cout << "Average norm of difference between identity and Kdd * inverse_Kdd: " << mean << endl << endl;
	}



/*	// initialization
	// dictionary
	MatrixXd d_t1(2,5);
	d_t1 << 1.6, 2.8, .5, -.7, 2.8,
			-3.2, .2, -.5, 1.7, .2;
    VectorXd new_sample = d_t1.col(d_t1.cols()-1);
	// weights
	VectorXd w_t1(5);
	w_t1 << -.4, .1, .3, -.7, .2;
	// kernel
	gaussian_kernel gk(.5);
	// _KDD
	MatrixXd _KDD;
	_KDD = gk.gram_matrix(d_t1,d_t1);
	// _KDD_inverse
	MatrixXd _KDD_inverse;
	_KDD_inverse = add_cols_to_inverse(_KDD, _KDD_inverse, d_t1, &gk);
	// error
	double err_max = .1;
	// print some stuff
	cout << "initial dictionary_t+1:\n" << d_t1 << endl; 
	cout << "new_sample:\n" << new_sample << endl;
	cout << "dict w/o 2nd val via remove_col_from_dict:\n" << remove_col_from_dict(d_t1, 1) << endl;
	MatrixXd d = d_t1;
	VectorXd w = w_t1;
	for(int j = 0; j < d.cols(); ++j){
		map<int, double> err_map;
		map<int, VectorXd> beta_map;
		// compute the if each value was excluded
		cout << "**************\nj = " << j << "\nd:\n" << d << "\nw:\n" << w << endl; 
		for(int i = 0; i < d.cols(); i++){
			cout << "\ni: " << i << endl;
			MatrixXd d_temp = remove_col_from_dict(d, i);
			MatrixXd _KDD_inverse_temp = remove_col_from_inverse(_KDD_inverse, i);
			// cout << "w_t1.transpose()[" << w_t1.transpose().rows() << "," << w_t1.transpose().cols() <<
			// 		"]*gram_matrix[" << gk.gram_matrix(d_t1,d_temp).rows() << "," << gk.gram_matrix(d_t1,d_temp).cols() <<
			// 		"]*_KDD_inverse[" << _KDD_inverse.rows() << "," << _KDD_inverse.cols() << "]\n";
			VectorXd beta = .5 * (w_t1.transpose() * gk.gram_matrix(d_t1,d_temp) * _KDD_inverse_temp);
			// approx_f = approx_f/(1+approx_f);
			double function_diff = w_t1.transpose() * gk.gram_matrix()
			double err = pow(error,2);
			cout << "d_temp:\n" << d_temp << endl;
			cout << "err: " << err << endl;
			err_map.insert({i, err});
			beta_map.insert({i, beta});
		}
		// find the i that gave the smallest error
		double err_best = 100000;
		int best_i = -1;
		for(int i = 0; i < d.cols(); i++){
			auto err_search = err_map.find(i);
			if(err_search->second < err_best){
				best_i = err_search->first;
				err_best = err_search->second;
			}
		}
		if (err_best > err_max){
			cout << "err_best: " << err_best << " not better than err_max: " << err_max << endl;
			break;
		}
		cout << "best_i: " << best_i << endl;
		cout << "err_best: " << err_best << endl;
		// update
		d = remove_col_from_dict(d, best_i);
		auto beta_search = beta_map.find(best_i);
		w = beta_search->second;
		_KDD_inverse = remove_col_from_inverse(_KDD_inverse, best_i);
	}*/
}