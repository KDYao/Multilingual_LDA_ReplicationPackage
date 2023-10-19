#!/usr/bin/python
try:
    from setuptools import setup, find_packages
except ImportError:
    try:
        from setuptools.core import setup, find_packages
    except ImportError:
        from distutils.core import setup, find_packages

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rest')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

required = open('requirements.txt').read().splitlines()
required = [l.strip() for l in required
            if l.strip() and not l.strip().startswith('#')]

setup(
    name='multilingual_lda',
    version='1.0',
    description='Multilingual LDA model on Bleeping Computers',
    license='MIT',
    author='Kundi Yao, Muhammad Parvez, Gustavo Oliva',
    packages=find_packages(),
    long_description=read_md('README.md'),
    install_requires=required
)
