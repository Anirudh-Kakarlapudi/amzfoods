import os
from setuptools import setup, find_packages

repo_dir = os.path.abspath(__file__)
long_description = 'Read README.md for long description'

setup(name='amzfoods',
      version='1.0',
      packages = find_packages()
      description='Amazonfinefoods',
      long_description = long_description
      author='Anirudh Kakarlapudi',
      author_email='mauryakak@gmail.com',
      url = '',
      classifiers = [
            'Topic:: Natural Language Processing::Sentiment Analysis',
            'Programming Language :: Python :: 3.6'
      ]
      python_requires='>=3.6'
     )