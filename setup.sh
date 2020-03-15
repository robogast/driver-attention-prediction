#!/bin/bash

set -e

# Get script and tmpdata dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TMPDATADIR="$TMPDIR"/data

# Prepare and unzip BDDA
echo "Please download BDDA.zip manually and place in root folder"

mkdir "$TMPDATADIR"
cp $DIR/BDDA.zip "$TMPDATADIR"
unzip "$TMPDATADIR"/BDDA.zip '*' -d "$TMPDATADIR" # unzip subfolders into $TMPDATADIR

ls "$TMPDATADIR" # for debugging purposes

# Get models
models_zip="pretrained_models.zip"
wget -nc --directory-prefix="$DIR" --no-check-certificate 'https://docs.google.com/uc?export=download&id=1q_CgyX73wrYTAsZjDF9aMXNPURcUmWVy' -O $models_zip
unzip -u -d "$DIR" $models_zip
rm $models_zip

# Get alexnet weights
wget -nc --directory-prefix="$DIR" https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy

$DIR/prepare_data.sh

cp -r "$TMPDATADIR"/*  "$DIR"/data