"""
Installs flatlands, a gym-compatible driving-on-track simulator
"""

from os import path
from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gym_flatlands',
    install_requires=[
        'gym',
        'scipy',
        'mpi4py',
        'cloudpickle',
        'tensorflow',
        'pyproj',
        'pygame',
    ],
    description='Flatlands simple on-track driving simulator with physics',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='AscentAI',
    url='https://github.com/ascentai/flatlands-gym',
    author_email='dev@ascent.ai',
    version='0.1.2',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='driving simulation gym',  # Optional,
    license='MIT',
    python_requires='>=3',
    data_files=[('my_data', ['gym_flatlands/envs/flatlands_sim/original_circuit_green.csv'])],
)
