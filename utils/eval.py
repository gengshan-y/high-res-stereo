from .readpfm import readPFM
import sys
from matplotlib import pyplot as plt
import glob
import cv2
import numpy as np
import os
from texttable import Texttable
import subprocess
import pdb
import time
import PIL.Image
from subprocess import call
import errno 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    return iter(p.stdout.readline, b'')

def eval_mb(eval_dir, subset, rdir,res, method_name, img_name, th,has_mask=False):
    # res relative to F
    with open('%s/%sF/%s/calib.txt'%(eval_dir,subset,img_name)) as f: 
        lines = f.readlines()
        rd = int(lines[7].split('=')[-1])
        max_disp = int(int(lines[6].split('=')[-1])*res)
    command = 'code/evaldisp %s/OO_OO%s/XX_XX/disp0%s.pfm %s/OO_OOF/XX_XX/disp0GT.pfm %f %d %d'%\
                (eval_dir, rdir,method_name, eval_dir,th, max_disp, rd)
    if has_mask:
        command += ' OO_OOF/XX_XX/mask0nocc.png'
    
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    return lines


def eval_carla(result_dir,method_name, img_name, th):
    image_suffix = img_name.split('/',4)[-1].replace('/','_').split('.')[0]
    command = 'code/evaldisp %s/%s%s.pfm /data/gengshay/carla-new-eval/%s.pfm %f 768 0'%\
                (result_dir,method_name,image_suffix,\
                 image_suffix.replace('cam0','depth0'), th)
    command = command.replace('XX_XX',img_name)
    lines = [i for i in run_command(command.split())]
    return lines

def run_iResNet(rdir,res):
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    os.chdir('/home/gengshay/code/iResNet/models') 
    command = 'python test_rob.py iResNet_ROB.caffemodel %f'%(res*2/1.5)
    lines = [i for i in run_command(command.split())]

    for filename in glob.glob('submission_results/Middlebury2014_iResNet_ROB/val/*'):
        imname=filename.split('/')[-1].strip()
        if os.path.isdir(filename):
            call(['cp', '-rf', 'submission_results/Middlebury2014_iResNet_ROB/val/%s'%imname, '/home/gengshay/code/MiddEval3/val%s/'%(rdir)])
    

def run_elas_carla(result_dir, res, img_name):
    max_disp = 768*res
    im = cv2.imread(img_name)
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    cv2.imwrite('./tmp0.png',im)
    im = cv2.imread(img_name.replace('cam0','cam1'))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    cv2.imwrite('./tmp1.png',im)
   
    image_suffix = img_name.split('/',4)[-1].replace('/','_').split('.')[0]
    command = 'alg-ELAS/run ./tmp0.png ./tmp1.png %d %s' % (max_disp,result_dir)
    lines = [i for i in run_command(command.split())]
    print (lines)

    command = 'mv %s/disp0.pfm %s/%s.pfm '%(result_dir,result_dir,image_suffix)
    lines = [i for i in run_command(command.split())]
    print (lines)

def run_elas(eval_dir, subset,rdir, res, img_name):
# res: resolution to upscale, not resolution compared to Full size
    mkdir_p('%s/%s%s/%s'%(eval_dir,subset,rdir,img_name))
    with open('%s/%sF/%s/calib.txt'%(eval_dir,subset,img_name)) as f:
        lines = f.readlines()
        max_disp = int(int(lines[6].split('=')[-1]) * res)
    if rdir == 'H':
        max_disp = max_disp//2
    elif rdir == 'Q':
        max_disp = max_disp//4
    if rdir=='A': 
        #max_disp = max_disp * res
        indir='F'
    else: indir = rdir
    im = cv2.imread('%s/%s%s/%s/im0.png'%(eval_dir,subset,indir,img_name))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    cv2.imwrite('./tmp0.png',im)
    im = cv2.imread('%s/%s%s/%s/im1.png'%(eval_dir,subset,indir,img_name))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    cv2.imwrite('./tmp1.png',im)
   
    command = 'alg-ELAS/run ./tmp0.png ./tmp1.png %d %s/%s%s/%s/' % (max_disp,eval_dir,subset,rdir,img_name)
    lines = [i for i in run_command(command.split())]
    print (lines)
 
    command = 'mv %s/OO_OO%s/XX_XX/time.txt %s/OO_OO%s/XX_XX/timeELAS.txt'%(eval_dir,rdir,eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

    command = 'mv %s/OO_OO%s/XX_XX/disp0_s.pfm %s/OO_OO%s/XX_XX/disp0ELAS_s.pfm'%(eval_dir,rdir,eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

    command = 'mv %s/OO_OO%s/XX_XX/disp0.pfm %s/OO_OO%s/XX_XX/disp0ELAS.pfm'%(eval_dir,rdir,eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)


def run_sgm(eval_dir, subset, rdir,res, img_name):
    with open('%s/%sF/%s/calib.txt'%(eval_dir,subset,img_name)) as f:
        lines = f.readlines()
        max_disp = int(int(lines[6].split('=')[-1]) * res)
    if rdir == 'H':
        max_disp = max_disp//2
    elif rdir == 'Q':
        max_disp = max_disp//4
    if rdir=='A': indir='F'
    else: indir = rdir
    im = cv2.imread('%s/%s%s/%s/im0.png'%(eval_dir,subset,indir,img_name))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    im = np.pad(im, ((max_disp*2,max_disp*2),(max_disp*2,max_disp*2),(0,0)),mode='constant')
    cv2.imwrite('%s/%s%s/%s/im0tmp.png'%(eval_dir,subset,rdir,img_name), im)
    im = cv2.imread('%s/%s%s/%s/im1.png'%(eval_dir,subset,indir,img_name))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    im = np.pad(im, ((max_disp*2,max_disp*2),(max_disp*2,max_disp*2),(0,0)),mode='constant')
    cv2.imwrite('%s/%s%s/%s/im1tmp.png'%(eval_dir,subset,rdir,img_name), im)

    # prepare images
    command = 'SGM/app %s/OO_OO%s/XX_XX/im0tmp.png %s/OO_OO%s/XX_XX/im1tmp.png -dst_path=%s/OO_OO%s/XX_XX/disp0tmp.png -max_disparity=%d -no-downscale' % (eval_dir,rdir,eval_dir,rdir,eval_dir,rdir,max_disp)
    #command = 'SGM/spsstereo OO_OO%s/XX_XX/im0tmp.png OO_OO%s/XX_XX/im1tmp.png' % (rdir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    begt = time.time()
    lines = [i for i in run_command(command.split())]
    totalt = time.time()-begt
    with open('%s/%s%s/%s/timeSGM.txt'%(eval_dir,subset,rdir,img_name),'w') as f:
        f.write(str(totalt))
    print (lines)
  
    # save pfm
    disp = np.asarray(PIL.Image.open('%s/%s%s/%s/disp0tmp.png'%(eval_dir,subset,rdir,img_name)))[max_disp*2:-max_disp*2,max_disp*2:-max_disp*2]
    #disp = np.asarray(PIL.Image.open('im0tmp_left_disparity.png'))
    with open('%s/%s%s/%s/disp0SGM.pfm'%(eval_dir,subset,rdir,img_name),'w') as f:
        save_pfm(f, disp.astype(np.float32)[::-1], scale=1./max_disp)
    #t1,t2=readPFM('%s%s/%s/disp0SGM.pfm'%(subset,res,img_name))
    command = 'rm %s/OO_OO%s/XX_XX/disp0tmp.png'%(eval_dir,rdir)
    #command = 'rm im0tmp_left_disparity.png'
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

    command = 'rm %s/OO_OO%s/XX_XX/im0tmp.png'%(eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

    command = 'rm %s/OO_OO%s/XX_XX/im1tmp.png'%(eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

def run_mccnn(eval_dir, subset, rdir,res, img_name):
    with open('%s/%sF/%s/calib.txt'%(eval_dir,subset,img_name)) as f:
        lines = f.readlines()
        max_disp = int(int(lines[6].split('=')[-1]) * res)
    if rdir == 'H':
        max_disp = max_disp//2
    elif rdir == 'Q':
        max_disp = max_disp//4
    if rdir=='A': indir='F'
    else: indir = rdir
    im = cv2.imread('%s/%s%s/%s/im0.png'%(eval_dir,subset,indir,img_name))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    #im = np.pad(im, ((max_disp*2,max_disp*2),(max_disp*2,max_disp*2),(0,0)),mode='constant')
    cv2.imwrite('%s/%s%s/%s/im0tmp.png'%(eval_dir,subset,rdir,img_name), im)
    im = cv2.imread('%s/%s%s/%s/im1.png'%(eval_dir,subset,indir,img_name))
    im = cv2.resize(im,None,fx=res,fy=res,interpolation=cv2.INTER_AREA)
    #im = np.pad(im, ((max_disp*2,max_disp*2),(max_disp*2,max_disp*2),(0,0)),mode='constant')
    cv2.imwrite('%s/%s%s/%s/im1tmp.png'%(eval_dir,subset,rdir,img_name), im)

    # prepare images
    os.chdir('mc-cnn')
    command = './main.lua mb slow -a predict -net_fname net/net_mb_slow_-a_train_all.t7 -left %s/OO_OO%s/XX_XX/im0tmp.png -right %s/OO_OO%s/XX_XX/im1tmp.png -disp_max %d' % (eval_dir,rdir,eval_dir,rdir,max_disp)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    begt = time.time()
    lines = [i for i in run_command(command.split())]
    totalt = time.time()-begt
    with open('%s/%s%s/%s/timeMCCNN.txt'%(eval_dir,subset,rdir,img_name),'w') as f:
        f.write(str(totalt))
    print (lines)

    command = 'luajit samples/bin2png.lua %d %d %d'%(max_disp,im.shape[0],im.shape[1])
    lines = [i for i in run_command(command.split())]
    print (lines)
  
    # save pfm
    #disp = np.asarray(PIL.Image.open('disp.png'))[max_disp*2:-max_disp*2,max_disp*2:-max_disp*2]
    disp = np.asarray(PIL.Image.open('disp.png')).astype(float)/2
#    disp = np.asarray(PIL.Image.open('left.png')).astype(float)
#    disp = disp/disp.max() * max_disp
    with open('%s/%s%s/%s/disp0MCCNN.pfm'%(eval_dir,subset,rdir,img_name),'w') as f:
        save_pfm(f, disp.astype(np.float32)[::-1], scale=1./max_disp)
    os.chdir('../')
    command = 'rm %s/OO_OO%s/XX_XX/im0tmp.png'%(eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

    command = 'rm %s/OO_OO%s/XX_XX/im1tmp.png'%(eval_dir,rdir)
    command = command.replace('XX_XX',img_name)
    command = command.replace('OO_OO',subset)
    lines = [i for i in run_command(command.split())]
    print (lines)

'''
Save a Numpy array to a PFM file.
'''
def save_pfm(file, image, scale = 1):
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)  
