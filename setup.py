from setuptools import setup
import os

CURRENT_DIR = os.path.dirname(__file__)
__VERSION__ = 0.1

setup(name='dlmark',
      version=__VERSION__,
      description=open(os.path.join(CURRENT_DIR, 'README.md')).read(),
      packages=['dlmark'],
      url='https://github.com/dmlc/dlmark')
