#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pip
from setuptools import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()


def parse_requirements(filename):
    lines = (line.strip() for line in open(filename))
    return [line for line in lines if line and not line.startswith('#')]


parsed_requirements = parse_requirements('requirements/prod.txt')
parsed_test_requirements = parse_requirements('requirements/test.txt')


setup(
    name='diluvian',
    version='0.0.6',
    description="Flood filling networks for segmenting electron microscopy of neural tissue.",
    long_description=readme + '\n\n' + history,
    author="Andrew S. Champion",
    author_email='andrew.champion@gmail.com',
    url='https://github.com/aschampion/diluvian',
    packages=[
        'diluvian',
    ],
    package_dir={'diluvian':
                 'diluvian'},
    entry_points={
        'console_scripts': [
            'diluvian=diluvian.__main__:main'
        ]
    },
    include_package_data=True,
    install_requires=parsed_requirements,
    license="MIT license",
    zip_safe=False,
    keywords='diluvian',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    setup_requires=['pytest-runner',],
    test_suite='tests',
    tests_require=parsed_test_requirements
)
