test: module
	cp ml_project*.so tests
	make -C tests
module:
	python setup.py build_ext
clean:
	rm ml_project.so
