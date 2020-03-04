# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 23:49:39 2017

@author: pasca
"""
import os
import argparse
from functools import partial
from multiprocessing import Pool

import imageio
import numpy as np
from tqdm import tqdm

def split_video(video_name, video_dir, image_dir, parse_rate, transform_fn, shift):
    filename = os.path.join(video_dir, video_name)
    video_id = video_name.split('.')[0]
    try:
        reader = imageio.get_reader(filename)
    except OSError:
        with open("video_parsing_errors.txt", "a") as myfile:
            myfile.write(video_id+'\n')
        return
    fps = reader.get_meta_data()['fps']
    duration = reader.get_meta_data()['duration']
    n_frames = reader.get_meta_data()['nframes']
    
    if parse_rate is not None:
        # calculate the time points in ms to sample frames
        time_points = np.arange(shift*1000, duration*1000, 1000.0/parse_rate)
        time_points = np.floor(time_points).astype(int)
        # calculate the frame indexes
        frame_indexes = (np.floor(time_points/1000.0*fps)).astype(int)
        sample_size = len(frame_indexes)
    else:
        frame_indexes = np.arange(n_frames)
        time_points = (frame_indexes*1000/fps).astype(int)
        sample_size = n_frames
    
    for i in range(sample_size):
        # make output file name
        image_name = os.path.join(image_dir, video_id+'_'+\
            str(time_points[i]).zfill(5)+'.jpg')
        
        # read image
        try:
            image = reader.get_data(frame_indexes[i])
        except:
            print('Can\'t read this frame. Skip')
            continue
        
        # apply transformation
        if transform_fn is not None:
            image = transform_fn(image)
        
        # write image
        imageio.imwrite(image_name, image)


def parse_videos(video_dir, image_dir, parse_rate, transform_fn=None, 
                 overwrite=False, shift=0, video_suffix='.mp4'):
    # parse_rate is how many Hz the videos should be parsed
    # shift is in seconds, means starting parsing in 'shift' seconds
    # overwrite=False means if some videos have already had parsed frames in image_dir, then skip these videos
    
    # make sure output directory is there
    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    
    video_names_set = set(f for f in os.listdir(video_dir) if f.endswith(video_suffix))

    if not overwrite:
        # collect already parsed videos
        old_video_ids = set(f.split('_')[0] for f in os.listdir(image_dir) if f.endswith('.jpg'))
        video_names_set &= old_video_ids
    
    video_names = list(video_names_set)

    with Pool(os.cpu_count() - 1 or 1) as pool:
        num = len(video_names)
        func = partial(
            split_video, video_dir=video_dir, image_dir=image_dir,
            parse_rate=parse_rate, transform_fn=transform_fn, shift=shift
        )
        for _ in tqdm(pool.imap_unordered(func, video_names), total=num):
            pass

    

def main(args):
    parse_rate = (args.sample_rate
                  if args.sample_rate % args.prediction_rate == 0
                  else args.sample_rate * args.prediction_rate)

    # parse videos
    parse_videos(
        args.video_dir, args.image_dir, overwrite=args.overwrite,
        parse_rate=parse_rate, video_suffix=args.video_suffix
    )
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',
        type=str,
        default='data/application/camera_videos',
        help='the directory that contains videos to parse')
    parser.add_argument('--image_dir',
        type=str,
        default='data/application/camera_images',
        help='the directory of parsed frame images')
    parser.add_argument('--sample_rate',
        type=int,
        default=3,
        help='at how many Hz the attention prediction results are needed')
    parser.add_argument('--prediction_rate',
        type=int,
        default=3,
        help='at how many Hz will the network predicts attention maps')
    parser.add_argument('--video_suffix',
        type=str,
        default='.mp4',
        help='the suffix of video files. E.g., .mp4')
    parser.add_argument('--overwrite',
        action='store_true',
        help='overwrite previously generated frames')
    
    args = parser.parse_args()
    
    main(args)
    

