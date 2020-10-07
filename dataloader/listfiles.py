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
    left_train = []
    right_train = []
    disp_train_l = []
    disp_train_r = []
    
    _, scenes, _ = list(os.walk(filepath))[0]
    print(scenes)
    for scene in scenes:
        _, coords, _ = list(os.walk(os.path.join(filepath, scene)))[0]
        print(coords)
        for coord in coords:
            _, pairs, _ = list(os.walk(os.path.join(filepath, *[scene, coord])))[0]
            print(pairs)
            left_train += [os.path.join(filepath, *[scene, coord, pair, 'im0.png']) for pair in pairs]
            right_train += [os.path.join(filepath, *[scene, coord, pair, 'im1.png']) for pair in pairs]
            disp_train_l += [os.path.join(filepath, *[scene, coord, pair, 'disp0GT.pfm']) for pair in pairs]
            disp_train_r += [os.path.join(filepath, *[scene, coord, pair, 'disp1GT.pfm']) for pair in pairs]
    return left_train, right_train, disp_train_l, disp_train_r
