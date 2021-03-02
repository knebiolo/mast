from setuptools import setup

setup(name = 'biotas',
      version = '0.0.3',
      description = '''BIO-Telemetry Analysis Software (BIOTAS) for use in
      removing false positive and overlap detections from radio telemetry projects.''',
      url = 'https://github.com/knebiolo/biotas',
      author = 'Kevin P. Nebiolo',
      author_email = 'kevin.nebiolo@kleinschmidtgroup.com',
      license = 'MIT',
      packages = ['biotas',],
      python_requires= '>=3.5',
      install_requires=["numpy >= 1.17.4",
                        "pandas >= 0.25.3",
                        "matplotlib >= 3.1.1",
                        "statsmodels >= 0.10.1",
                        "networkx >= 2.2",
                        "sqlite"],
      zip_safe = False
      )
