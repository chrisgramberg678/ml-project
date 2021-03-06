from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import eigency

extensions = [
    Extension(name="ml_project",
    		  sources=["ml_project.pyx", "kernel.cpp", "model.cpp", "gradient_descent.cpp"],
              extra_compile_args=['-std=c++11'],
              include_dirs= eigency.get_includes(include_eigen=True),
              language="c++",
              )
]

setup(
	ext_modules = cythonize(extensions)
)