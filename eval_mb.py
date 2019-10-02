import warnings
warnings.filterwarnings("ignore")
import numpy as np
import glob
from matplotlib import pyplot as plt
from texttable import Texttable
from utils.readpfm import readPFM
import cv2
import pdb
import os
import matplotlib.patches as patches

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str ,default='HSM',
                    help='resolution')
parser.add_argument('--indir', type=str ,default='/ssd/carla-highres/carlaF/',
                    help='resolution')
parser.add_argument('--gtdir', type=str ,default='/ssd/carla-highres/testF/',
                    help='resolution')
args = parser.parse_args()

method = args.method


t = Texttable()
t.set_deco(Texttable.HEADER)
t.set_cols_dtype(['t','f','f','f','f','f','f'])
t.set_cols_width([32,10,10,10,10,10,10])
t.add_row(['Image', 'avgerr','rms','bad-1','bad-2','bad-4','time'])


imgnames=[i for i in sorted(glob.glob('%s/*'%args.indir)) if os.path.isdir(i)]
avgerrs = np.zeros(len(imgnames)+1)
rmss = np.zeros(len(imgnames)+1)
bad5s = np.zeros(len(imgnames)+1)
bad10s = np.zeros(len(imgnames)+1)
bad20s = np.zeros(len(imgnames)+1)
times = np.zeros(len(imgnames)+1)

for i,imgname in enumerate(imgnames):
    if not os.path.isdir(imgname):
        continue
    gt = readPFM('%s/%s/disp0GT.pfm'%(args.gtdir,imgname.split('/')[-1]))[0]
    mask = gt!=np.inf
    disp = readPFM('%s/disp0%s.pfm'%(imgname,method))[0]
    if disp.shape != gt.shape:
        ratio = float(gt.shape[1])/disp.shape[1]
        disp=cv2.resize(disp,(gt.shape[1],gt.shape[0]))*ratio

    with open('%s/%s/calib.txt'%(args.gtdir,imgname.split('/')[-1])) as f:
        lines = f.readlines()
        max_disp = int(lines[6].split('=')[-1])
    disp[disp>max_disp]=max_disp

    errmap = np.abs(gt-disp)*mask
    avgerr = errmap[mask].mean()
    rms = np.sqrt((errmap[mask]**2).mean())
    bad5map = (errmap>1) * mask
    bad5 = bad5map[mask].sum()/float(mask.sum())*100

    bad10map = (errmap>2) * mask
    bad10 = bad10map[mask].sum()/float(mask.sum())*100

    bad20map = (errmap>4) * mask
    bad20 = bad20map[mask].sum()/float(mask.sum())*100

    avgerrs[i] = avgerr
    rmss[i] = rms
    bad5s[i] = bad5
    bad10s[i] = bad10
    bad20s[i] = bad20
    with open('%s/time%s.txt'%(imgname,method)) as f:
        times[i]=float(f.readlines()[0])

# remove the Shopvac image
avgerrs[-1]= avgerrs[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].mean()
rmss[-1] =      rmss[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].mean()
bad5s[-1] =    bad5s[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].mean()
bad10s[-1] =   bad10s[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].mean()
bad20s[-1] =   bad20s[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].mean()
times[-1] =    times[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]].mean()
#avgerrs[-1]= avgerrs.mean()
#rmss[-1] =    rmss.mean()
#bad5s[-1] =    bad5s.mean()
#bad10s[-1] =    bad10s.mean()
#bad20s[-1] =    bad20s.mean()
#times[-1] =    times.mean()
t.add_rows(zip([i.split('/')[-1] for i in imgnames]+['ALL'],avgerrs,rmss,bad5s,bad10s,bad20s,times),header=False)


t.set_cols_align(['r','r','r','r','r','r','r'])
print (t.draw())
