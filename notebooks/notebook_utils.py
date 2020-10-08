import os
from io import BytesIO

import boto3
import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.preprocess import get_inv_transform

CLIENT = boto3.client('s3')


def get_lidar_train_list():
    data = BytesIO()
    CLIENT.download_fileobj('autogpe-datasets', 'lidar-hdsm-dataset/train_1_0.txt', data)
    data.seek(0)
    json_str = data.read().decode('UTF-8')
    return json_str.split('\r\n')


def list_s3(prefix):
    objs = CLIENT.list_objects_v2(Bucket='autogpe-datasets', Prefix=prefix, Delimiter='/')
    return objs


def download_sample(s3_key, out_folder):
    command = 'aws s3 sync "{}" "{}" --quiet'.format(s3_key, out_folder)
    print(command)
    os.system(command)
    print('===== Finished dataset sync =====')
    return out_folder


def visualize_random_pixel_pairs(left, right, disp_left, sample=4):
    ys = []
    for i in range(sample):
        y = np.random.randint(0, disp_left.shape[0], 1)[0]
        x = np.random.randint(0, disp_left.shape[1], 1)[0]
        disp = disp_left[y, x]
        while disp < 1e-6:
            y = np.random.randint(0, disp_left.shape[0], 1)
            x = np.random.randint(0, disp_left.shape[1], 1)
            disp = disp_left[y, x]
        cv2.circle(left, (x, y), 10, [255, 0, 0])
        cv2.circle(right, (int(x - disp), y), 10, [0, 255, 0])
        cv2.circle(right, (x, y), 10, [0, 0, 255])
        ys.append(y)

    rectified_pair = np.concatenate((left, right), axis=1)
    h, w, _ = rectified_pair.shape
    for i in ys:
        rectified_pair = cv2.line(rectified_pair, (0, i), (w, i), (0, 0, 255))
    return rectified_pair


def visualize_sample(img_l, img_r, disp_l):
    preprocess_transform = get_inv_transform()
    img_l = preprocess_transform(img_l)
    img_r = preprocess_transform(img_r)
    img_l = (np.array(img_l).transpose((1, 2, 0)) * 255).astype(np.uint8)
    img_r = (np.array(img_r).transpose((1, 2, 0)) * 255).astype(np.uint8)

    rect_pair = visualize_random_pixel_pairs(img_l, img_r, disp_l)

    normalized_disparity_image = cv2.normalize(disp_l, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    normalized_disparity_image = np.stack(
        [normalized_disparity_image, normalized_disparity_image, normalized_disparity_image]).transpose((1, 2, 0))
    disp_overlay = cv2.addWeighted(img_l, 0.4, normalized_disparity_image, 0.8, 0)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 35))
    ax1.set_title('Disparity Overlay')
    ax1.imshow(disp_overlay)
    ax2.imshow(normalized_disparity_image)
    ax2.set_title('Disparity')
    ax3.imshow(rect_pair)
    ax3.set_title('Pair')

    fig.show()
