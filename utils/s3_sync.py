import os
from concurrent.futures import ThreadPoolExecutor

import boto3


def sync_open_dataset(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    command = 'aws s3 sync "s3://autogpe-datasets/hdsm" "{}" --quiet'.format(dataset_dir)
    print(command)
    os.system(command)
    print('===== Finished open datasets sync =====')


def sync_airsim_dataset(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    command = 'aws s3 sync "s3://autogpe-datasets/airsim_dataset" "{}/airsim" --quiet'.format(dataset_dir)
    print(command)
    os.system(command)
    print('===== Finished airsim sync =====')


def sync_lidar_dataset(dataset_dir):
    bucket_name = 'autogpe-datasets'
    s3 = boto3.resource('s3')
    target_dir = f'{dataset_dir}/lidar-hdsm-dataset'
    os.makedirs(target_dir, exist_ok=True)

    # download validation file
    val_file = f'{target_dir}/val_1_0.txt'
    s3.meta.client.download_file(bucket_name, 'lidar-hdsm-dataset/val_1_0.txt', val_file)

    # sync all the regions in parallel
    regions = {
        'Demo-Dallas': [1,2,3],
        'Demo-LA-River': [1,2],
        'Demo-Manchester': [1,2,3],
        'Demo-Richmond': list(range(1, 14)),
        'Demo-Seattle': [1,2,3],
    }

    tlds = []
    for region, subregions in regions.items():
        tlds.extend(['{}-{}'.format(region, subregion) for subregion in subregions])

    commands = [f'aws s3 sync "s3://{bucket_name}/lidar-hdsm-dataset/{tld}" "{target_dir}/{tld}"' for tld in tlds]
    with ThreadPoolExecutor(os.cpu_count() // 2) as executor:
        executor.map(os.system, commands)

    # build train/val sets
    all_files = []
    for path, subdirs, files in os.walk(dataset_dir):
        files = [f for f in files if f]
        for name in files:
            all_files.append(os.path.join(path, name).replace("\\","/"))
    train_set = set([os.path.dirname(f).replace(f'{target_dir}/', '') for f in all_files if os.path.basename(f) == 'im0.png'])
    with open(val_file) as f:
        val_set = set([line.replace('s3://autogpe-datasets/lidar-hdsm-dataset/', '').strip() for line in f.readlines()])

    # remove items from val_set that don't exist
    val_set = val_set.intersection(train_set)

    # remove items from train_set that are in val_set
    train_set = list(train_set.difference(val_set))
    val_set = list(val_set)
    with open(os.path.join(target_dir, 'train.txt'), 'w') as f:
        f.writelines([f'{i}/n' for i in train_set])
    with open(os.path.join(target_dir, 'val.txt'), 'w') as f:
        f.writelines([f'{i}/n' for i in val_set])
    return target_dir, train_set, val_set


def sync_dataset(dataset_dir):
    sync_open_dataset(dataset_dir)
    sync_airsim_dataset(dataset_dir)
    sync_lidar_dataset(dataset_dir)


def sync_open_dataset_small(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    command = 'aws s3 sync "s3://autogpe-datasets/hdsm_small" "{}" --quiet'.format(dataset_dir)
    print(command)
    os.system(command)
    print('===== Finished hdsm_small dataset sync =====')


if __name__ == '__main__':
    sync_lidar_dataset(r'C:/tmp')
