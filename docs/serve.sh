#!/bin/bash

#Serves the documentation on localhost:8080
#Builds if necessary

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

#Check if the build directory exists
if [ ! -d "_build" ]; then
    ./build.sh
fi

cd _build/html

echo "Docs available at http://localhost:8080"
python3 -m http.server -b 127.0.0.1 8080 > /dev/null 2>&1 