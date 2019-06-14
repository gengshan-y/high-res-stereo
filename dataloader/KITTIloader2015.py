import torch.utils.data as data

import pdb
from PIL import Image
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, typ = 'train'):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'

  image = [img for img in os.listdir(filepath+left_fold) if img.find('_10') > -1]
  image = sorted(image)
  imglist = [1,3,6,20,26,35,38,41,43,44,49,60,67,70,81,84,89,97,109,119,122,123,129,130,132,134,141,144,152,158,159,165,171,174,179,182, 184,186,187,196]
  if typ == 'train':
    train = [image[i] for i in range(200) if i not in imglist]*100
  elif typ == 'trainval':
    train = [image[i] for i in range(200)]*80
  val = [image[i] for i in imglist]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]
  disp_train_L = [filepath+disp_L+img for img in train]

  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]
  disp_val_L = [filepath+disp_L+img for img in val]

  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L
