"""
Installs flatlands, a gym-compatible driving-on-track simulator
"""

from setuptools import setup

setup(name='gym_flatlands',
      install_requires=[
          'gym',
          'scipy',
          'mpi4py',
          'cloudpickle',
          'tensorflow',
          'pygame',
          'pyproj'
      ],
      description='Flatlands simple on-track driving simulator with physics',
      author='AscentAI',
      url='https://github.com/ascentai/flatlands-gym',
      author_email='david@ascent.ai',
      version='0.1.2')
