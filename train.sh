#!/bin/bash

set -e

DATADIR="$TMPDIR"/data
INPATH="/project/robertjs/data"
OUTPATH="./logs/retrained_base_model"

cp -r "$INPATH" "$DATADIR"

python train.py \
--data_dir="$DATADIR" \
--model_dir="$OUTPATH" \
--batch_size=10 \
--n_steps=6 \
--feature_name=alexnet \
--train_epochs=500 \
--epochs_before_validation=3 \
--image_size 288 512 \
--feature_map_channels=256 \
--quick_summary_period=20 \
--slow_summary_period=100