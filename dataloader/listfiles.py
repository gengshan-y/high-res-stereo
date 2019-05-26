import torch.utils.data as data

import pdb
from PIL import Image
import os
import os.path
import numpy as np
import glob


def dataloader(filepath):
  img_list = ['Adirondack',  'Motorcycle',    'PianoL',
              'Playtable',   'Shelves',       'ArtL',
              'MotorcycleE', 'Pipes',         'PlaytableP',
              'Teddy',       'Jadeplant',     'Piano',
              'Playroom',    'Recycle',       'Vintage']

  img_list = ['Australia',   'Bicycle2',   'Classroom2E',
              'Crusade',   'Djembe',   'Hoops',       'Newkuba',
              'Staircase', 'AustraliaP',  'Classroom2',  'Computer',     
              'CrusadeP',  'DjembeL',  'Livingroom',  'Plants']

  img_list = [i.split('/')[-1] for i in glob.glob('%s/*'%filepath) if os.path.isdir(i)]
  #img_list *= 10

  left_train  = ['%s/%s/im0.png'% (filepath,img) for img in img_list]  
  right_train = ['%s/%s/im1.png'% (filepath,img) for img in img_list]
  disp_train_L = ['%s/%s/disp0GT.pfm' % (filepath,img) for img in img_list]
  disp_train_R = disp_train_L

  return left_train, right_train
