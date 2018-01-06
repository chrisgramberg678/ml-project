test: module
	mv ml_project.so tests
	cd tests
	# python tests/test_kernel.py
	# python tests/test_batch_gradient_descent.py
	python tests/test_stochastic_gradient_descent.py
module:
	python setup.py build_ext
clean:
	rm grad.so
