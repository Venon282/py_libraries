# py_libraries

## update the codocumentation
py -m pip install sphinx furo sphinx-autodoc-typehints numpydoc
cd docs
sphinx-apidoc -o source/ ../../py_libraries --module-first --force
py -m pip install sphinx-rtd-theme
./make html