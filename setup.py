from setuptools import setup

setup(name = 'pymast',
      version = '0.0.1',
      description = '''Movement Analysis Software for Telemetry (MAST) for
use in removing false positive and overlap detections from radio telemetry
projects and assessing 1D movement patterns.''',
      url = 'https://github.com/knebiolo/mast',
      author = 'Kevin P. Nebiolo and Theodore Castro-Santos',
      author_email = 'kevin.nebiolo@kleinschmidtgroup.com',
      license = 'MIT',
      packages = ['pymast',],
      python_requires= '>=3.5',
      install_requires=["numpy >= 1.17.4",
                        "pandas >= 0.25.3",
                        "matplotlib >= 3.1.1",
                        "statsmodels >= 0.10.1",
                        "networkx >= 2.2",
                        "scipy >= 1.7.1",
                        "sklearn",
                        "h5py"],
      zip_safe = False
      )
