# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:48:31 2020

@author: Kevin Nebiolo
"""

from setuptools import setup

setup(name = 'biotas',
      version = '0.0.1',
      description = '''BIO-Telemetry Analysis Software (BIOTAS) for use in 
      removing false positive and overlap detections from radio telemetry projects.''',
      url = 'https://github.com/knebiolo/biotas',
      author = 'Kevin P. Nebiolo',
      author_email = 'kevin.nebiolo@kleinschmidtgroup.com',
      license = 'MIT',
      packages = ['biotas',],
      python_requires= '>=3.6',
      zip_safe = False
      )