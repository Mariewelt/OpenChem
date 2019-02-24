# Build the python package, don't let setuptools/pip try to get packages
pip install . --no-deps
if errorlevel 1 exit 1
