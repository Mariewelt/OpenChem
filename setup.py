"""
OpenChem -- Deep Learning toolkit for Computational Chemistry and Drug Design
"""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_attrs = {
    'name': 'OpenChem',
    'description': 'Deep Learning toolkit for Computational Chemistry and Drug Design with PyTorch backend',
    'long_description': long_description,
    'long_description_content_type': "text/markdown",
    'url': 'https://github.com/Mariewelt/OpenChem',
    'author': 'Mariewelt',
    'author_email': 'mariewelt@gmail.com',
    'license': 'MIT',
    'packages': find_packages(),
    'include_package_data': True,
    'use_scm_version': True,
    'setup_requires': ['setuptools_scm'],
    'install_requires': [
        'tensorboard',
        'networkx',
        'tqdm',
        'torchani'
    ],
    'python_requires': ">=3.5",
    'zip_safe': False
}

setup(**setup_attrs)
