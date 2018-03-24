#include "omp.cpp"
/*
	Test file for getting omp working before adding it into any models
	run with: g++ -std=c++11 -I /usr/local/include/eigen3 kernel.cpp test_omp.cpp -o omp && ./omp
 */

bool test_add_col_to_dict(){
	MatrixXd m(2,4);
	m << 1, 2, 3, 4,
		 5, 6, 7, 8;
	VectorXd v(2);
	v << 2, 2;
	m = add_col_to_dict(m, v);
	MatrixXd m2(2,5);
	m2 << 1, 2, 3, 4, 2,
		  5, 6, 7, 8, 2;
	return m.isApprox(m2);
}

bool test_remove_col_from_dict(){
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
	if(verbose && norm >= threshold){
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

// tests remove_col_from_inverse() by removing each column from dictionary and checking that the resulting inverse is valid
// this is done by checking ((Kdd_inverse * Kdd) - MatrixXd::Identity()).norm() < threshold for each kernel matrix and associated inverse computed 
// by excluding a single value from the dictionary
// this depends on remove_col_from_dict working properly
bool remove_col_from_inverse_test(int rows, int cols, double threshold, std::vector<double> &norms, bool verbose=false){
	gaussian_kernel gk(.5);

	std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd{0,5};
    bool vals[rows];

	MatrixXd d_t1(rows, cols);

	// initialize the dictionary randomly
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j)
			d_t1(i,j) = nd(gen);
	}

	// compute a gram matrix and inverse for t1
	MatrixXd Kdd_t1 = gk.gram_matrix(d_t1,d_t1);
	MatrixXd inverse_Kdd_t1 = add_cols_to_inverse(Kdd_t1, MatrixXd(), d_t1, &gk);
	// compute the gram matrix and inverse for each column of d_t1 being removed
	for(int c = 0; c < cols; ++c){
		MatrixXd d_t = remove_col_from_dict(d_t1, c);
		MatrixXd Kdd_t = gk.gram_matrix(d_t, d_t);
		MatrixXd Kdd_t_inverse = remove_col_from_inverse(inverse_Kdd_t1, c);
		MatrixXd approx_identity = Kdd_t * Kdd_t_inverse;
		MatrixXd identity = MatrixXd::Identity(Kdd_t.rows(),Kdd_t.cols());
		// the norm of the difference between the approx_identity and the the identity should be small
		MatrixXd diff = approx_identity - identity;
		double norm = diff.norm();
		norms.push_back(norm);
		if(verbose && norm >= threshold){
			stringstream info;
			info << "***************************************************************\n";
			info << "Dictonary - Rows: " << rows << ". Cols: " << cols << " to " << cols - 1 << ".\n";
			info << d_t1 << endl << endl;
			info << "removing col: " << c << endl;
			info << d_t << endl << endl;
			info << "Kdd_t:\n" << Kdd_t << endl << endl;
			info << "Inverse:\n" << Kdd_t_inverse << endl << endl;
			info << "Eigen inverse:\n" << Kdd_t.inverse() << endl << endl;
			info << "Kdd * Kdd_t_inverse:\n" << approx_identity << endl << endl;
			info << "Difference from identity:\n" << diff << endl << endl;
			info << "Norm < threshold: " << norm << " < " << threshold << " -> " << ((norm < threshold) ? "True" : "False") << endl; 
			info << "***************************************************************\n";
			cout << info.str() << endl;
		}
		vals[c] = norm < threshold;
	}
	bool return_val = true;
	for(auto v : vals){
		return_val &= v;
	}
	return return_val;
}

bool test_inverting_kernel_matrices(){
	int samples = 30;
	int tries = 200;
	std::vector<bool> results;
	for(int s = 2; s < samples; ++s){
		std::vector<double> norms;
		for(int t = 0; t < tries; ++t){
			results.push_back(invert_Kdd_test(2, s, .00001, norms, true));
		}
		double mean = 0;
		for(auto n : norms){
			mean += n;
		}
		mean/=tries;
		// cout << "Samples: " << s << endl;
		// cout << "Constructed Inverse: Average norm of difference between identity and Kdd * inverse_Kdd: " << mean << endl << endl;
	}

	for(int s = 2; s < samples; ++s){
		std::vector<double> norms;
		for(int t = 0; t < tries; ++t){
			results.push_back(remove_col_from_inverse_test(2, s, .00001, norms, true));
		}
		double mean = 0;
		for(auto n : norms){
			mean += n;
		}
		mean/=norms.size();
		// cout << "Samples: " << s << endl;
		// cout << "Removed a col Inverse: Average norm of difference between identity and Kdd * inverse_Kdd: " << mean << endl << endl;
	}
	bool result = true;
	for(auto r : results){
		result &= r;
	}
	return result;
}

// unit test for when we have identical columns in KDD
// call it KDD_stable and make a big matrix to add and remove to arbitraily
bool inverting_inverse_with_duplicates_test(int rows, int cols, double threshold, bool verbose=true){
	// create a large dictionary with some random duplicates
	gaussian_kernel gk(.2);

	std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> nd{0,5};

	MatrixXd d(rows, cols);

	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j)
			d(i,j) = nd(gen);
	}

	// make sure some cols are duplicates
	for(int i = 0; i < cols; ++i){
		if(rand() % 10 == 0){
			d.col(i) = d.col(rand() % cols);
		}
	}

	// build the initial gram matrix and inverse
	MatrixXd Kdd = gk.gram_matrix_stable(d,d);
	MatrixXd Kdd_inverse = add_cols_to_inverse(Kdd, MatrixXd(), d, &gk);
	// randomly add and remove cols and check to see if the norm is lower than threhold
	if(verbose){
		cout << "before the loop\n";
		stringstream ss;
		ss << "initial dictionary:\n" << d << endl << endl;
		ss << "initial Kdd:\n" << Kdd << endl << endl;
		ss << "initial inverse:\n" << Kdd_inverse << endl << endl;
		ss << "eigen inverse:\n" << Kdd.inverse() << endl << endl;
		cout << ss.str();
	}	
	for(int i = 0; i < 200; ++i){
		cout << "*******************************************************************************************************\n";
		int r = rand();
		// add or remove
		if(true/*(r % 3 || d.cols() < 3)*/){
			cout << "adding ";
			// add something new
			if(false){
				VectorXd v(rows);
				v << nd(gen), nd(gen);
				cout << "new value:\n";// << v << endl;
				d = add_col_to_dict(d, v);
			}
			// add a duplicate
			else{
				cout << "duplicate:\n";// << d.col(r % d.cols()) << endl;
				d = add_col_to_dict(d, d.col(r % d.cols()));
			}
			// update Kdd
			Kdd = add_samples_to_Kdd(Kdd, d, &gk);
			// update inverse
			Kdd_inverse = add_cols_to_inverse(Kdd, Kdd_inverse, d, &gk);
		}
		else{
			//remove
			cout << "removing col " << r % d.cols() << ":\n" << d.col(r % d.cols()) << endl;
			d = remove_col_from_dict(d, r % d.cols());
			Kdd = remove_sample_from_Kdd(Kdd, r % d.cols());
			Kdd_inverse = remove_col_from_inverse(Kdd_inverse, r % d.cols());
		}
		// find the norm
		MatrixXd diff = (Kdd * Kdd_inverse) - MatrixXd::Identity(Kdd.rows(), Kdd.cols());
		if(verbose){
			stringstream loop;
			loop << "New dictionary:\n" << d << endl << endl;
			loop << "New Kdd:\n" << Kdd << endl << endl;
			loop << "Kdd_inverse:\n" << Kdd_inverse << endl << endl;
			loop << "Eigen inverse:\n" << Kdd.inverse() << endl << endl; 
			loop << "Kdd * Kdd_inverse(this should look like the identity):\n" << (Kdd * Kdd_inverse) << endl << endl;
			cout << loop.str();
			cout << Kdd.rows() << "," << Kdd.cols() << " * " << Kdd_inverse.rows() << "," << Kdd_inverse.cols() << endl;
			cout << "find the norm\n";
			// string a;
			// cin >> a;
		}
		cout << diff.norm() << " > " << threshold << " is " << ((diff.norm() > threshold) ? "True" : "False") << endl; 
		if(diff.norm() > threshold){
			return false;
		}
	}
	return true;
}

bool test_inverting_with_duplicates(){
	return inverting_inverse_with_duplicates_test(2, 4, 100);
}

int main(){
	// cout << "adding columns to samples works: " << (test_add_col_to_dict() ? "True" : "False") << endl;
	// cout << "removing columns from dict works: " << (test_remove_col_from_dict() ? "True" : "False") << endl;
	// cout << "inverting matrices works: " << (test_inverting_kernel_matrices() ? "True" : "False") << endl;
	cout << "inverting with duplicates works: " << (test_inverting_with_duplicates() ? "True" : "False") << endl;

	// initialization
	// dictionary
	/*MatrixXd d_t1(2,4);
	d_t1 << 1, 2, 3, 1,
			4, 4, 6, 4;
    VectorXd new_sample = d_t1.col(d_t1.cols()-1);
	// weights
	VectorXd w_t1(4);
	w_t1 << -.4, .1, .3, -.7;
	// kernel
	gaussian_kernel gk(.1);
	// _KDD
	MatrixXd _KDD = gk.gram_matrix_stable(d_t1,d_t1);
	// _KDD_inverse
	MatrixXd _KDD_inverse = add_cols_to_inverse(_KDD, MatrixXd(), d_t1, &gk);
	// error
	double err_max = 1e-6;
	// print some stuff
	cout << "initial dictionary_t+1:\n" << d_t1 << endl << endl; 
	cout << "initial Kernel matrix:\n" << _KDD << endl << endl;
	cout << "initial inverse Kernel matrix:\n" << _KDD_inverse << endl << endl;
	cout << "should be similar to identity:\n" << _KDD * _KDD_inverse << endl << endl;
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
			VectorXd beta = (w.transpose() * gk.gram_matrix(d, d_temp) * _KDD_inverse_temp);
			double residual_error = (w.transpose() * gk.gram_matrix(d, d) * w).value() + 
									(beta.transpose() * gk.gram_matrix(d_temp, d_temp) * beta).value() - 
									2 * (w.transpose() * gk.gram_matrix(d, d_temp) * beta).value();
			double first_term = (w.transpose() * gk.gram_matrix(d, d) * w).value();
			double second_term = (beta.transpose() * gk.gram_matrix(d_temp, d_temp) * beta).value();
			double third_term = 2 * (w.transpose() * gk.gram_matrix(d, d_temp) * beta).value();
			cout << "error = " << first_term << " + " << second_term << " - " <<  third_term << endl;
			cout << "d_temp:\n" << d_temp << endl;
			cout << "beta:\n" << beta << endl;
			cout << "residual_error: " << residual_error << endl;
			err_map.insert({i, residual_error});
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