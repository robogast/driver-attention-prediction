set -e

python parse_videos.py \
--video_dir=data/training/camera_videos \
--image_dir=data/training/camera_images \
--sample_rate=3 \
--video_suffix=.mp4

python parse_videos.py \
--video_dir=data/training/gazemap_videos \
--image_dir=data/training/gazemap_images \
--sample_rate=3 \
--video_suffix=.mp4

python parse_videos.py \
--video_dir=data/validation/camera_videos \
--image_dir=data/validation/camera_images \
--sample_rate=3 \
--video_suffix=.mp4

python parse_videos.py \
--video_dir=data/validation/gazemap_videos \
--image_dir=data/validation/gazemap_images \
--sample_rate=3 \
--video_suffix=.mp4

python write_tfrecords_for_inference.py \
--data_dir=data/training \
--n_divides=2 \
--longest_seq=35

python write_tfrecords_for_inference.py \
--data_dir=data/validation \
--n_divides=2 \
--longest_seq=35

python make_feature_maps.py \
--data_dir=data/training \
--model_dir=pretrained_models/model_for_inference

python make_feature_maps.py \
--data_dir=data/validation \
--model_dir=pretrained_models/model_for_inference

python write_tfrecords.py \
--data_dir=data/training \
--n_divides=2 \
--feature_name=alexnet \
--image_size 288 512 \
--longest_seq=35

python write_tfrecords.py \
--data_dir=data/training \
--n_divides=2 \
--feature_name=alexnet \
--image_size 288 512 \
--longest_seq=35