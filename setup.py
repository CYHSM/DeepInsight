"""
DeepInsight Toolbox
© Markus Frey
https://github.com/CYHSM/DeepInsight
Licensed under MIT License
"""
from setuptools import setup

long_description = open('README.md').read()
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='deepinsight',
    version='0.5',
    install_requires=requirements,
    author='Markus Frey',
    author_email='markus.frey1@gmail.com',
    description="A general framework for interpreting wide-band neural activity",
    long_description=long_description,
    url='https://github.com/CYHSM/DeepInsight/',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)
