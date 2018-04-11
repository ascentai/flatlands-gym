"""
Installs flatlands, a gym-compatible driving-on-track simulator
"""

from os import path
from setuptools import setup

# Get the long description from the README file
HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    LONG_DESC = f.read()

setup(
    name='flatlands',
    install_requires=[
        'gym',
        'scipy',
        'pyproj',
    ],
    description='Flatlands simple on-track driving simulator with physics',
    long_description=LONG_DESC,
    long_description_content_type='text/markdown',
    author='AscentAI',
    url='https://github.com/ascentai/flatlands-gym',
    author_email='dev@ascent.ai',
    version='0.1.3',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='driving simulation gym',  # Optional,
    license='MIT',
    python_requires='>=3',
    data_files=[('map', ['flatlands/envs/flatlands_sim/original_circuit_green.csv'])],
    packages=['flatlands'],
)
