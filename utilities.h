/*
 * This file contains all the headers that the project should need as well as some handy helpers for interfacing to Cython
 */
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include "math.h"
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// helpers for interfacting to Cython
vector<double> eigen_to_stl(VectorXd v);
vector< vector<double> > eigen_to_stl(MatrixXd m);
VectorXd stl_to_eigen(vector<double> v);
MatrixXd stl_to_eigen(vector< vector<double> > v);