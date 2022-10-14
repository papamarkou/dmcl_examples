from codecs import open
from os import path
from setuptools import find_packages, setup

from dmcl_examples import __version__

url = 'https://bitbucket.org/papamarkou/dmcl_examples'

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='dmcl_examples',
    version=__version__,
    description='Experiments of deep Markov chaing learning',
    long_description=long_description,
    url=url,
    download_url='{0}/get/master.tar.gz'.format(url),
    packages=find_packages(),
    license='MIT',
    author='Theodore Papamarkou',
    author_email='theodore.papamarkou@gmail.com',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'
    ],
    keywords=['Bayesian', 'MCMC', 'Monte Carlo', 'neural networks'],
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.2',
        'pandas',
        'scikit-learn',
        'torch>=1.9.0',
        'torchvision>=0.9.1',
        'eeyore>=0.0.20',
        'kanga>=0.0.20',
        'matplotlib>=3.3.3',
        'seaborn>=0.11.0',
        'ruptures>=1.1.3'
    ],
    package_data={'dmcl_examples': ['data/pima.csv', 'data/*/x.csv', 'data/*/y.csv', 'data/*/readme.md']},
    include_package_data=True,
    zip_safe=False
)
