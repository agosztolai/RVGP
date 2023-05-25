from setuptools import setup, find_packages

import numpy
from Cython.Build import cythonize
from setuptools import Extension

requirements = (
   'cython==0.29.35',
   'numpy==1.23.5',
   'scipy==1.10.1',
   'matplotlib==3.7.1',
   'gpflow==2.6.5',
   'tensorflow==2.12.0',
   'tensorflow-probability==0.20.0',
   'scikit-learn==1.2.2',
   'networkx==3.1'
)

setup(name='RVGP',
      version='0.1',
      packages=find_packages(exclude=["examples*"]),
      python_requires='>=3.6',
      install_requires=requirements,
      package_data={"RVGP.lib": ["ptu_dijkstra.pyx", "ptu_dijkstra.c"]},
      ext_modules=cythonize(
          Extension(
              "ptu_dijkstra", ["RVGP/lib/ptu_dijkstra.pyx"], include_dirs=[numpy.get_include()]
          )
      ),
      )
