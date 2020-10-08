import os
import os.path


def dataloader(filepath):
    _, dirs, _ = list(os.walk(filepath))[0]
    left_train = ['%s/%s/im0.png' % (filepath, sample_dir) for sample_dir in dirs]
    right_train = ['%s/%s/im1.png' % (filepath, sample_dir) for sample_dir in dirs]
    disp_train_l = ['%s/%s/disp0GT.pfm' % (filepath, sample_dir) for sample_dir in dirs]
    disp_train_r = ['%s/%s/disp1GT.pfm' % (filepath, sample_dir) for sample_dir in dirs]
    return left_train, right_train, disp_train_l, disp_train_r


def airsim_dataloader(filepath):
    left_train = []
    right_train = []
    disp_train_l = []
    _, scenes, _ = list(os.walk(filepath))[0]
    for scene in scenes:
        _, orientations, _ = list(os.walk(os.path.join(filepath, scene)))[0]
        for orientation in orientations:
            left_train.append(os.path.join(filepath, *[scene, orientation, 'left_bgr.png']))
            right_train.append(os.path.join(filepath, *[scene, orientation, 'right_bgr.png']))
            disp_train_l.append(os.path.join(filepath, *[scene, orientation, 'left_disparity.npy']))
    return left_train, right_train, disp_train_l


def lidar_dataloader(filepath):
    train_file = f'{filepath}/train.txt'
    with open(train_file) as f:
        lines = [l.strip() for l in f.readlines()]
    samples = [os.path.join(filepath, l) for l in lines]
    im0 = [f'{sample}/im0.png' for sample in samples]
    im1 = [f'{sample}/im1.png' for sample in samples]
    disp0 = [f'{sample}/disp0GT.pfm' for sample in samples]
    disp1 = [f'{sample}/disp1GT.pfm' for sample in samples]
    return im0, im1, disp0, disp1

