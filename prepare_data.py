#!/usr/bin/env python

import subprocess
from argparse import ArgumentParser
from pathlib import Path


def call_parse_videos(video_dir, image_dir):
    sample_rate = 3
    video_suffix = '.mp4'
    
    subprocess.run((
        'python parse_videos.py'
        f' --video_dir {video_dir}'
        f' --image_dir {image_dir}'
        f' --sample_rate {sample_rate}'
        f' --video_suffix {video_suffix}'
    ).split(), check=True)
    

def call_write_tfrecords_for_inference(data_dir):
    n_divides = 2
    longest_seq = 35
    
    subprocess.run((
        'python write_tfrecords_for_inference.py'
        f' --data_dir {data_dir}'
        f' --n_divides {n_divides}'
        f' --longest_seq {longest_seq}'
    ).split(), check=True)


def call_make_feature_maps(data_dir, model_dir):
    subprocess.run((
        'python make_feature_maps.py'
        f' --data_dir {data_dir}'
        f' --model_dir {model_dir}'
    ).split(), check=True)


def call_write_tfrecords(data_dir):
    n_divides = 2
    longest_seq = 35
    feature_name = 'alexnet'
    image_size = '288 512'
    
    subprocess.run((
        'python write_tfrecords.py'
        f' --data_dir {data_dir}'
        f' --n_divides {n_divides}'
        f' --feature_name {feature_name}'
        f' --image_size {image_size}'
        f' --longest_seq {longest_seq}'
    ).split(), check=True)


def prepare_data(data_dir, subcategories, model_dir):
    for subcategory in subcategories:
        call_parse_videos(data_dir / (subcategory + '_videos'),
                          data_dir / (subcategory + '_images'))
    
    call_write_tfrecords_for_inference(data_dir)
    call_make_feature_maps(data_dir, model_dir)
    call_write_tfrecords(data_dir)


def main(data_dir, subfolders, subcategories, model_dir):

    data_dir, model_dir = map(lambda x: x.resolve(), (data_dir, model_dir))

    for subfolder in subfolders:
        prepare_data(data_dir / subfolder, subcategories, model_dir)


if __name__ == '__main__':
    current_path = Path(__file__).parent
    parser = ArgumentParser(description='prepare data for predicting driver attention training code')
    parser.add_argument('-d', '--data-dir', type=Path, default=(current_path / 'data'))
    parser.add_argument('-m', '--model-dir', type=Path, default=(current_path / 'pretrained_models'))
    parser.add_argument('--subfolders', nargs='+', default=['training', 'validation'])
    parser.add_argument('--subcategories', nargs='+', default=['camera', 'gazemap'],
                        help='a category folder should end with `_videos`, e.g.: `camera_videos`')
    args = parser.parse_args()

    main(**vars(args))


# Get script dir
# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# DATADIR="$TMPDIR"/data

# python parse_videos.py \
# --video_dir="$DATADIR"/training/camera_videos \
# --image_dir="$DATADIR"/training/camera_images \
# --sample_rate=3 \
# --video_suffix=.mp4

# python parse_videos.py \
# --video_dir="$DATADIR"/training/gazemap_videos \
# --image_dir="$DATADIR"/training/gazemap_images \
# --sample_rate=3 \
# --video_suffix=.mp4

# python parse_videos.py \
# --video_dir="$DATADIR"/validation/camera_videos \
# --image_dir="$DATADIR"/validation/camera_images \
# --sample_rate=3 \
# --video_suffix=.mp4

# python parse_videos.py \
# --video_dir="$DATADIR"/validation/gazemap_videos \
# --image_dir="$DATADIR"/validation/gazemap_images \
# --sample_rate=3 \
# --video_suffix=.mp4

# python write_tfrecords_for_inference.py \
# --data_dir="$DATADIR"/training \
# --n_divides=2 \
# --longest_seq=35

# python write_tfrecords_for_inference.py \
# --data_dir="$DATADIR"/validation \
# --n_divides=2 \
# --longest_seq=35

# python make_feature_maps.py \
# --data_dir="$DATADIR"/training \
# --model_dir=pretrained_models/model_for_inference

# python make_feature_maps.py \
# --data_dir="$DATADIR"/validation \
# --model_dir=pretrained_models/model_for_inference

# python write_tfrecords.py \
# --data_dir="$DATADIR"/training \
# --n_divides=2 \
# --feature_name=alexnet \
# --image_size 288 512 \
# --longest_seq=35

# python write_tfrecords.py \
# --data_dir="$DATADIR"/validation \
# --n_divides=2 \
# --feature_name=alexnet \
# --image_size 288 512 \
# --longest_seq=35