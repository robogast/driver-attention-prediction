#!/bin/bash

set -e

# Get script dir
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DATADIR="$TMPDIR"/data

cd $DIR

python parse_videos.py \
--video_dir="$DATADIR"/training/camera_videos \
--image_dir="$DATADIR"/training/camera_images \
--sample_rate=3 \
--video_suffix=.mp4

python parse_videos.py \
--video_dir="$DATADIR"/training/gazemap_videos \
--image_dir="$DATADIR"/training/gazemap_images \
--sample_rate=3 \
--video_suffix=.mp4

python parse_videos.py \
--video_dir="$DATADIR"/validation/camera_videos \
--image_dir="$DATADIR"/validation/camera_images \
--sample_rate=3 \
--video_suffix=.mp4

python parse_videos.py \
--video_dir="$DATADIR"/validation/gazemap_videos \
--image_dir="$DATADIR"/validation/gazemap_images \
--sample_rate=3 \
--video_suffix=.mp4

python write_tfrecords_for_inference.py \
--data_dir="$DATADIR"/training \
--n_divides=2 \
--longest_seq=35

python write_tfrecords_for_inference.py \
--data_dir="$DATADIR"/validation \
--n_divides=2 \
--longest_seq=35

python make_feature_maps.py \
--data_dir="$DATADIR"/training \
--model_dir=pretrained_models/model_for_inference

python make_feature_maps.py \
--data_dir="$DATADIR"/validation \
--model_dir=pretrained_models/model_for_inference

python write_tfrecords.py \
--data_dir="$DATADIR"/training \
--n_divides=2 \
--feature_name=alexnet \
--image_size 288 512 \
--longest_seq=35

python write_tfrecords.py \
--data_dir="$DATADIR"/training \
--n_divides=2 \
--feature_name=alexnet \
--image_size 288 512 \
--longest_seq=35