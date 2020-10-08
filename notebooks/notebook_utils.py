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
    print('===== Finished hdsm_small dataset sync =====')
    return out_folder


def visualize_sample(data_loader):
    preprocess_transform = get_inv_transform()
    for (img_l, img_r, disp_l) in data_loader:
        img_l = preprocess_transform(img_l)
        img_r = preprocess_transform(img_r)
        img_l = (np.array(img_l).transpose((1, 2, 0)) * 255).astype(np.uint8)
        img_r = np.array(img_r).transpose((1, 2, 0))

        normalized_disparity_image = cv2.normalize(disp_l, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        normalized_disparity_image = np.stack(
            [normalized_disparity_image, normalized_disparity_image, normalized_disparity_image]).transpose((1, 2, 0))
        disp_overlay = cv2.addWeighted(img_l, 0.5, normalized_disparity_image, 0.8, 0)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(20, 35))
        ax1.set_title('Disparity Overlay')
        ax1.imshow(disp_overlay)
        ax2.imshow(normalized_disparity_image)
        ax3.imshow(img_l)
        ax4.imshow(img_r)
        fig.show()
    return
