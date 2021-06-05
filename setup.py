from distutils.core import setup

setup(
    name='PollutionSelect',
    version='0.1',
    packages=['pollution_select',],
    license='MIT',
    long_description=open('README.rst').read(),
    install_requires=[
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "seaborn",
    ]
)