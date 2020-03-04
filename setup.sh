#!/bin/bash

set -e

# Get script dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# TODO get BDDA
echo "Please download BDDA manually"

# Get alexnet weights
wget -nc --directory-prefix=$DIR https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy

$DIR/prepare_data.sh