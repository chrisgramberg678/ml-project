test: module
	mv grad.so tests
	cd tests
	python tests/test_kernel.py
	python tests/test_batch_kernel_regression.py
module:
	python gradient_descent_setup.py build_ext --inplace
clean:
	rm *.so
	rm *.o