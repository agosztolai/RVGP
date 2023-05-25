from setuptools import setup, find_packages

import numpy
from Cython.Build import cythonize
from setuptools import Extension

# requirements = (
   # 'numpy>=1.19.1',
  # 'tensorflow>=2.2.0',
  # 'gpflow>=2.0.5',
  # 'tensorflow-probability>=0.9.0',
# )

extra_requirements = {
  'examples': (
      # 'networkx',
      'matplotlib',
      'scipy',
      # 'tqdm',
      # 'spharapy',
      # 'stripy',
      # 'osmnx',
      # 'contextily',
      # 'shapely',
      # 'pandas',
      # 'warn',
      # 'tables',
  ),
}

setup(name='RVGP',
      version='0.1',
      packages=find_packages(exclude=["examples*"]),
      python_requires='>=3.6',
      # install_requires=requirements,
      extras_require=extra_requirements,
      package_data={"graph_matern.lib": ["ptu_dijkstra.pyx", "ptu_dijkstra.c"]},
      ext_modules=cythonize(
          Extension(
              "ptu_dijkstra", ["graph_matern/lib/ptu_dijkstra.pyx"], include_dirs=[numpy.get_include()]
          )
      ),
      )
