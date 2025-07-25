#!/bin/bash

# Builds the documentation

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# uncomment if tutorials are available
#mkdir -p tutorials
#cp ../examples/*.ipynb tutorials

rm -r API/
make clean
make html
