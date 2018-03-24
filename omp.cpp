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
		// the one column version of the inverse is a 1x1 matrix with the value (K_DD(0,0))^-1.
		if(c == 0){
			_KDD_inverse = MatrixXd(1,1);
			_KDD_inverse(0,0) = 1.0/_KDD(0,0);
		}
		// add a single row and column
		else{
			// create some intermediary values
			VectorXd u1 = _k->gram_matrix_stable(_dictionary.leftCols(c), _dictionary.col(c));
			VectorXd u2 = _KDD_inverse * u1;
			double d = 1/(_k->gram_matrix_stable(_dictionary.col(c), _dictionary.col(c)).value() - (u1.transpose() * u2).value());
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
	if(old_inverse.rows() != old_inverse.cols()){
		stringstream ss;
		ss << "Invalid inverse matrix. Number of rows and columns should match. Given: rows = " << old_inverse.rows() << ", cols = " << old_inverse.cols();
		throw std::invalid_argument(ss.str());
	}
	if(i > old_inverse.cols() - 1 || i < 0){
		stringstream ss;
		ss << "Cannot remove column " << i << " from inverse matrix with " << old_inverse.cols() << " columns.";
		throw std::invalid_argument(ss.str());
	}
	// permute the ith column and ith column to the last column and last row
	VectorXd ith_col = old_inverse.col(i);
	double ith_col_i = ith_col(i);
	for(int j = i + 1; j < old_inverse.cols(); ++j){
		old_inverse.row(j - 1) = old_inverse.row(j);
		old_inverse.col(j - 1) = old_inverse.col(j);
		ith_col(j - 1) = ith_col(j);
	}
	ith_col(ith_col.size() - 1) = ith_col_i;
	old_inverse.bottomRows(1) = ith_col.transpose();
	old_inverse.rightCols(1) = ith_col;
	// create some intermediary values
	MatrixXd top_left = old_inverse.topLeftCorner(old_inverse.rows() - 1, old_inverse.cols() - 1);
	double d = old_inverse(old_inverse.rows() - 1, old_inverse.rows() - 1);
	VectorXd u3 = -1 * old_inverse.topRightCorner(old_inverse.rows() - 1, 1);
	VectorXd u2 = u3/d;
	// build the new inverse out of those values
	MatrixXd new_inverse = top_left - (d * (u2 * u2.transpose()));
	return new_inverse;
}

MatrixXd remove_sample_from_Kdd(MatrixXd Kdd, int i){
	for(int j = i + 1; j < Kdd.cols(); ++j){
		Kdd.row(j - 1) = Kdd.row(j);
		Kdd.col(j - 1) = Kdd.col(j);
	}
	return Kdd.topLeftCorner(Kdd.rows() - 1, Kdd.cols() -1);
}

MatrixXd add_samples_to_Kdd(MatrixXd Kdd, MatrixXd d, kernel* k){
	int i = Kdd.cols();
	Kdd.conservativeResize(d.cols(), d.cols());
	for(; i < d.cols(); ++i){
		VectorXd new_col = k->gram_matrix_stable(d, d.col(i));
		Kdd.col(i) = new_col;
		Kdd.row(i) = new_col.transpose();
	}
	return Kdd;
}

MatrixXd remove_col_from_dict(MatrixXd d, int i){
	if(i > d.cols() - 1 || i < 0){
		stringstream ss;
		ss << "Cannot remove column " << i << " from dictionary of length " << d.cols() << ".";
		throw std::invalid_argument(ss.str());
	}
	// permute the target column to the end
	for(int j = i + 1; j < d.cols(); ++j){
		d.col(j - 1) = d.col(j);
	}
	// return all but the last column
	return d.leftCols(d.cols() - 1);
}

MatrixXd add_col_to_dict(MatrixXd d, VectorXd v){
	// the batch size
	// make room in the dictionary for the new samples
	if(d.cols() == 0){
		d = MatrixXd(v.rows(),1);
	}
	else{
		d.conservativeResize(d.rows(), d.cols() + 1);
	}
	// update the dictionary after computing the new weights
	d.col(d.cols() - 1) = v;
	return d;
}