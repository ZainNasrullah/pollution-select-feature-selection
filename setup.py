from distutils.core import setup

setup(
    name='PollutionSelect',
    version='0.1',
    packages=['pollution_select',],
    license='MIT',
    long_description=open('README.rst').read(),
    install_requires=[
    "matplotlib==3.1.1",
    "numpy==1.17.2",
    "pandas==0.25.1",
    "scikit-learn==0.21.3",
    "scipy==1.3.1",
    "seaborn==0.9.0",
    ]
)