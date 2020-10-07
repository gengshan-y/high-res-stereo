import os


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
    os.makedirs(dataset_dir, exist_ok=True)
    command = 'aws s3 sync "s3://autogpe-datasets/lidar-hdsm-dataset" "{}/lidar" --quiet'.format(dataset_dir)
    print(command)
    os.system(command)
    print('===== Finished lidar sync =====')


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
