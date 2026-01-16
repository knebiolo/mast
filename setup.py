from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pymast',
    version='1.0.0',
    description='Movement Analysis Software for Telemetry (MAST) - Complete solution for radio telemetry data analysis',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/knebiolo/mast',
    author='Kevin P. Nebiolo and Theodore Castro-Santos',
    author_email='kevin.nebiolo@kleinschmidtgroup.com',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "statsmodels>=0.12.0",
        "networkx>=2.5",
        "scipy>=1.7.1",
        "scikit-learn>=0.24.0",
        "h5py>=3.0.0",
        "dask>=2021.3.0",
        "dask-ml>=1.9.0",
        "distributed>=2021.3.0",
        "numba>=0.53.0",
        "tables>=3.8.0",
        "intervaltree>=3.1.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords='telemetry, radio telemetry, fish tracking, movement ecology, false positive detection',
    zip_safe=False
)
