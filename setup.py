import glob
from setuptools import setup, find_packages

opts = dict(scripts=glob.glob("scripts/*.py"))


if __name__ == '__main__':
    setup(**opts)
