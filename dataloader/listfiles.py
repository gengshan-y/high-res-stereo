import torch.utils.data as data

import pdb
from PIL import Image
import os
import os.path
import numpy as np
import glob


def dataloader(filepath):
  img_list = [i.split('/')[-1].split('\\')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]

  left_train  = ['%s/%s/im0.png'% (filepath,img) for img in img_list]  
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]
  disp_train_R = ['%s/%s/disp1GT.pfm' % (filepath,img) for img in img_list]

  return left_train, right_train, disp_train_L, disp_train_R
