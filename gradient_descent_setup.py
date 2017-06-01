from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("grad", ["grad.pyx"],
              extra_compile_args=['-std=c++11'])
]

setup(
	ext_modules = cythonize(extensions)
)
