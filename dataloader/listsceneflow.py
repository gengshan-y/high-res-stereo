import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    monkaa_path = os.path.join(filepath, 'monkaa/frames_cleanpass')
    monkaa_disp = os.path.join(filepath, 'monkaa/disparity')
    monkaa_dir = os.listdir(monkaa_path)

    all_left_img = []
    all_right_img = []
    all_left_disp = []
    all_right_disp = []

    for dd in monkaa_dir:
        for im in os.listdir(monkaa_path+'/'+dd+'/left/'):
            if is_image_file(monkaa_path+'/'+dd+'/left/'+im):
                all_left_img.append(monkaa_path+'/'+dd+'/left/'+im)
                all_left_disp.append(monkaa_disp+'/'+dd+'/left/'+im.split(".")[0]+'.pfm')
                all_right_disp.append(monkaa_disp+'/'+dd+'/right/'+im.split(".")[0]+'.pfm')

        for im in os.listdir(monkaa_path+'/'+dd+'/right/'):
            if is_image_file(monkaa_path+'/'+dd+'/right/'+im):
                all_right_img.append(monkaa_path+'/'+dd+'/right/'+im)

    driving_dir = os.path.join(filepath, 'driving/frames_cleanpass')
    driving_disp = os.path.join(filepath, 'driving/disparity')
    _, subdir1, _ = list(os.walk(driving_dir))[0]
    _, subdir2, _ = list(os.walk(os.path.join(driving_dir, subdir1[0])))[0]
    _, subdir3, _ = list(os.walk(os.path.join(driving_dir, *[subdir1[0], subdir2[0]])))[0]

    for i in subdir1:
        for j in subdir2:
            for k in subdir3:
                imm_l = os.path.join(driving_dir, *[i, j, k, 'left'])
                for im in imm_l:
                    im_l = os.path.join(driving_dir, *[i, j, k, 'left', im])
                    im_r = os.path.join(driving_dir, *[i, j, k, 'right', im])
                    disp_l = os.path.join(driving_disp, *[i, j, k, 'left', im.split(".")[0] + '.pfm'])
                    disp_r = os.path.join(driving_disp, *[i, j, k, 'right', im.split(".")[0] + '.pfm'])
                    if is_image_file(im_l) and is_image_file(im_r):
                        all_left_img.append(im_l)
                        all_right_img.append(im_r)
                        all_left_disp.append(disp_l)
                        all_right_disp.append(disp_r)
    return all_left_img, all_right_img, all_left_disp, all_right_disp
