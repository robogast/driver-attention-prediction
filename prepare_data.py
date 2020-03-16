from argparse import ArgumentParser
from pathlib import Path

async def call_parse_videos(video_dir, image_dir):
    sample_rate = 3
    video_suffix = '.mp4'
    
    await asyncio.create_subprocess_shell(
        'python parse_videos.py',
        f' --video_dir={video_dir}',
        f' --image_dir={image_dir}',
        f' --sample_rate={sample_rate}',
        f' --video_suffix={video_suffix}'
    )


async def call_write_tfrecords_for_inference(data_dir):
    n_divides = 2
    longest_seq = 35
    
    await asyncio.create_subprocess_shell(
        'python write_tfrecords_for_inference.py',
        f' --n_divides={n_divides}',
        f' --longest_seq={longest_seq}',
    )


async def call_make_feature_maps(data_dir, model_dir):
    await asyncio.create_subprocess_shell(
        'python make_feature_maps.py',
        f' --data_dir={data_dir}',
        f' --model_dir={model_dir}'
    )


async def call_write_tfrecords(data_dir):
    n_divides = 2
    longest_seq = 35
    feature_name = 'alexnet'
    image_size = '288 512'
    
    await asyncio.create_subprocess_shell(
        'python write_tfrecords.py',
        f' --data_dir={data_dir}',
        f' --n_divides={n_divides}',
        f' --feature_name={feature_name}',
        f' --image_size={image_size}',
        f' --longest_seq={longest_seq}'
    )


async def prepare_data(data_dir, subcategories, model_dir):
    await asyncio.gather(
        call_parse_videos(data_dir / (subcategory + '_videos'), data_dir / (subcategory + '_images'))
        for subcategory in subcategories
    )

    await call_write_tfrecords_for_inference(data_dir)
    await call_make_feature_maps(data_dir, model_dir)
    await call_write_tfrecords(data_dir)

async def start_async(gen_fn):
    await asyncio.gather(*gen_fn)

def main(data_dir, subfolders, subcategories, model_dir):
    data_dir, model_dir = map(Path, (data_dir, model_dir))
    asyncio.run(start_async((prepare_data(data_dir / subfolder, subcategories, model_dir) for subfolder in subfolders)))


if __name__ == '__main__':
    current_path = Path(__file__).parent
    parser = argparse.ArgumentParser(description='prepare data for predicting driver attention training code')
    parser.add_argument('-d', '--datapath', type=Path, default=current_path / 'data')
    parser.add_argument('-m', '--modelpath', type=Path, default=current_path / 'pretrained_models')
    parser.add_argument('--subfolders', nargs='+', default=['training', 'validation'])
    parser.add_argument('--subcategories', nargs='+', default=['camera', 'gazemap'],
                        help='a category folder should end with `_videos`, e.g.: `camera_videos`')
    args = parser.parse_args()

    main(
        data_dir=args.datapath, subfolders=args.subfolders,
        subcategories=args.subcategories, model_dir=args.model_path
    )


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