#!/bin/bash

# Build the python package, don't let setuptools/pip try to get packages
# $PYTHON setup.py develop --no-deps
pip install . --no-deps