#!/usr/bin/env python

import subprocess
from argparse import ArgumentParser
from pathlib import Path

def unzip_bdda(data_dir, bdda_path):

    subprocess.run((
        'bsdtar --strip-components=1 -xvf'
        f' {bdda_path}'
        f' -C {data_dir}'
    ).split(), check=True)

def get_pretrained_models(model_dir):
    models_zip_name = "pretrained_models.zip"
    model_zip_url = 'https://docs.google.com/uc?export=download&id=1q_CgyX73wrYTAsZjDF9aMXNPURcUmWVy'

    subprocess.run((
        'wget -nc'
        f' --directory-prefix={model_dir}'
        f' --no-check-certificate {model_zip_url}'
        f' -O {models_zip_name}'
    ).split(), check=True)

    subprocess.run((
        'unzip -u'
        f' {model_dir / models_zip_name}'
        f' -d {model_dir}'
    ).split(), check=True)

    subprocess.run((
        f'rm {model_dir / models_zip_name}'
    ).split(), check=True)


def get_alexnet_weights(model_dir):
    weights_url = "https://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy"
    
    subprocess.run((
        'wget -nc'
        f' --directory-prefix={model_dir}'
        f' {weights_url}'
    ).split(), check=True)


def main(data_dir, bdda_path, model_dir):
    data_dir, bdda_path, model_dir = map(lambda x: x.resolve(), (data_dir, bdda_path, model_dir))

    unzip_bdda(data_dir, bdda_path)
    get_pretrained_models(model_dir)
    get_alexnet_weights(model_dir)


if __name__ == '__main__':
    current_path = Path(__file__).parent
    parser = ArgumentParser(description='prepare data for predicting driver attention training code')
    parser.add_argument('-d', '--data-dir', type=Path, default=(current_path / 'data'))
    parser.add_argument('-m', '--model-dir', type=Path, default=current_path)
    parser.add_argument('-b', '--bdda-path', type=Path, default=current_path / 'BDDA.zip')
    args = parser.parse_args()

    main(**vars(args))