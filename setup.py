from setuptools import setup, find_packages

setup(
    name='pickPocket',
    packages=['pickPocket'],
    version='0.1.0',
    description='Setting up a python package for pickPocket',
    author='Luca Gagliardi',
    author_email='luca.gagliardi@iit.it'
    # package_dir={'pickPocket': 'src'}
    # packages = ['pickPocket'],
    #packages=find_packages(include=['exampleproject', 'exampleproject.*']),
    # install_requires=[
    #     'PyYAML',
    #     'pandas==0.23.3',
    #     'numpy>=1.14.5'
    # ],
    # extras_require={'plotting': ['matplotlib>=2.2.0', 'jupyter']},
    # setup_requires=['pytest-runner', 'flake8'],
    # tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': ['my-command=exampleproject.example:main']
    # },
    # package_data={'exampleproject': ['data/schema.json']}
)