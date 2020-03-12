#!/bin/bash

set -e

# Get script dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cp -r $DIR/data $TMP

OUTPATH="logs/retrained_base_model"

python train.py \
--data_dir=$TMP/data \
--model_dir=$TMP/$OUTPATH \
--batch_size=10 \
--n_steps=6 \
--feature_name=alexnet \
--train_epochs=500 \
--epochs_before_validation=3 \
--image_size 288 512 \
--feature_map_channels=256 \
--quick_summary_period=20 \
--slow_summary_period=100

cp -r $TMP/$OUTPATH $DIR/$OUTPATH