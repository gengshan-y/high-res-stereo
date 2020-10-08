from __future__ import print_function

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from dataloader import KITTIloader2012 as lk12
from dataloader import KITTIloader2015 as lk15
from dataloader import MiddleburyLoader as DA
from dataloader import listfiles as ls
from dataloader import listsceneflow as lt
from dataloader.listfiles import lidar_dataloader
from models import hsm
from utils import logger
from utils import sync_dataset, persist_saved_models

torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description='HSM-Net')
    parser.add_argument('--maxdisp', type=int, default=384, help='maxium disparity')
    parser.add_argument('--logname', default='logname', help='log name')
    parser.add_argument('--database', default='./data', help='data path')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batchsize', type=int, default=28, help='samples per batch')
    parser.add_argument('--loadmodel', default=None, help='weights path')
    parser.add_argument('--savemodel', default='./model', help='save path')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--no-sync-dataset', action='store_true', help='Do not sync the dataset files')
    parser.add_argument('--persist_to_s3', action='store_true', help='Sync the output models to s3')
    parser.add_argument('--experiment_name', type=str, default='default', help='experiment name when persisting model to s3')
    args = parser.parse_args()
    return args


def load_model(input_args):
    torch.manual_seed(input_args.seed)
    model = hsm(input_args.maxdisp, clean=False, level=1)
    model = nn.DataParallel(model)
    model.cuda()

    # load model
    if input_args.loadmodel is not None:
        pretrained_dict = torch.load(input_args.loadmodel)
        pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if ('disp' not in k)}
        model.load_state_dict(pretrained_dict['state_dict'], strict=False)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    torch.manual_seed(input_args.seed)  # set again
    torch.cuda.manual_seed(input_args.seed)
    return model, optimizer


def _init_fn(worker_id):
    np.random.seed()
    random.seed()


def init_dataloader(input_args):
    batch_size = input_args.batchsize
    scale_factor = input_args.maxdisp / 384.  # controls training resolution

    hrvs_folder = '%s/hrvs/carla-highres/trainingF' % input_args.database
    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(hrvs_folder)
    loader_carla = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                    rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=2)

    middlebury_folder = '%s/middlebury/mb-ex-training/trainingF' % input_args.database
    all_left_img, all_right_img, all_left_disp, all_right_disp = ls.dataloader(middlebury_folder)
    loader_mb = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp,
                                 rand_scale=[0.225, 0.6 * scale_factor], rand_bright=[0.8, 1.2], order=0)

    rand_scale = [0.9, 2.4 * scale_factor]
    all_left_img, all_right_img, all_left_disp, all_right_disp = lt.dataloader('%s/sceneflow/' % input_args.database)
    loader_scene = DA.myImageFloder(all_left_img, all_right_img, all_left_disp,
                                    right_disparity=all_right_disp, rand_scale=rand_scale, order=2)

    # change to trainval when finetuning on KITTI
    all_left_img, all_right_img, all_left_disp, _, _, _ = lk15.dataloader('%s/kitti15/training/' % input_args.database,
                                                                          split='train')
    loader_kitti15 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=rand_scale, order=0)

    all_left_img, all_right_img, all_left_disp = lk12.dataloader('%s/kitti12/training/' % input_args.database)
    loader_kitti12 = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=rand_scale, order=0)

    all_left_img, all_right_img, all_left_disp, _ = ls.dataloader('%s/eth3d/' % input_args.database)
    loader_eth3d = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, rand_scale=rand_scale, order=0)

    all_left_img, all_right_img, all_left_disp, all_right_disp = lidar_dataloader('%s/lidar-hdsm-dataset/' %input_args.database)
    loader_lidar = DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_disparity=all_right_disp, rand_scale=[0.5, 1.1*scale_factor], order=2)

    data_inuse = torch.utils.data.ConcatDataset([loader_carla] * 10 +
                                                [loader_mb] * 150 + # 71 pairs
                                                [loader_scene] +  # 39K pairs 960x540
                                                [loader_kitti15] +
                                                [loader_kitti12] * 24 +
                                                [loader_eth3d] * 300 +
                                                [loader_lidar] ) # 25K pairs
                                                                 # airsim ~750
    train_dataloader = torch.utils.data.DataLoader(data_inuse, batch_size=batch_size, shuffle=True,
                                                   num_workers=batch_size, drop_last=True, worker_init_fn=_init_fn)
    print('%d batches per epoch' % (len(data_inuse) // batch_size))
    return train_dataloader


def train(model, optimizer, maxdisp, img_l, img_r, disp_l):
    model.train()
    img_l = Variable(torch.FloatTensor(img_l))
    img_r = Variable(torch.FloatTensor(img_r))
    disp_l = Variable(torch.FloatTensor(disp_l))

    img_l, img_r, disp_true = img_l.cuda(), img_r.cuda(), disp_l.cuda()

    # ---------
    mask = (disp_true > 0) & (disp_true < maxdisp)
    mask.detach_()
    # ----

    optimizer.zero_grad()
    stacked, entropy = model(img_l, img_r)
    loss = (64. / 85) * F.smooth_l1_loss(stacked[0][mask], disp_true[mask], size_average=True) + \
           (16. / 85) * F.smooth_l1_loss(stacked[1][mask], disp_true[mask], size_average=True) + \
           (4. / 85) * F.smooth_l1_loss(stacked[2][mask], disp_true[mask], size_average=True) + \
           (1. / 85) * F.smooth_l1_loss(stacked[3][mask], disp_true[mask], size_average=True)
    loss.backward()
    optimizer.step()
    vis = {'output3': stacked[0].detach().cpu().numpy(),
           'output4': stacked[1].detach().cpu().numpy(),
           'output5': stacked[2].detach().cpu().numpy(),
           'output6': stacked[3].detach().cpu().numpy(),
           'entropy': entropy.detach().cpu().numpy()}
    loss_val = loss.data
    del stacked
    del loss
    return loss_val, vis


def adjust_learning_rate(optimizer, epoch, input_args):
    if epoch <= input_args.epochs - 1:
        lr = 1e-3
    else:
        lr = 1e-4
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    input_args = parse_args()
    if not input_args.no_sync_dataset:
        print('===== Syncing dataset =====')
        sync_dataset(input_args.database)
        print('===== Data synced =========')

    hdsm_model, optimizer = load_model(input_args)

    log = logger.Logger(input_args.savemodel, name=input_args.logname)
    total_iters = 0

    for epoch in range(1, input_args.epochs + 1):
        total_train_loss = 0
        adjust_learning_rate(optimizer, epoch, input_args)

        train_img_loader = init_dataloader(input_args)
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(train_img_loader):
            start_time = time.time()
            loss, vis = train(hdsm_model, optimizer, input_args.maxdisp, imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss

            if total_iters % 10 == 0:
                log.scalar_summary('train/loss_batch', loss, total_iters)
            if total_iters % 100 == 0:
                log.image_summary('train/left', imgL_crop[0:1], total_iters)
                log.image_summary('train/right', imgR_crop[0:1], total_iters)
                log.image_summary('train/gt0', disp_crop_L[0:1], total_iters)
                log.image_summary('train/entropy', vis['entropy'][0:1], total_iters)
                log.histo_summary('train/disparity_hist', vis['output3'], total_iters)
                log.histo_summary('train/gt_hist', np.asarray(disp_crop_L), total_iters)
                log.image_summary('train/output3', vis['output3'][0:1], total_iters)
                log.image_summary('train/output4', vis['output4'][0:1], total_iters)
                log.image_summary('train/output5', vis['output5'][0:1], total_iters)
                log.image_summary('train/output6', vis['output6'][0:1], total_iters)

            total_iters += 1

            if (total_iters + 1) % 2000 == 0:
                save_filename = os.path.join(input_args.savemodel, *[input_args.logname,
                                                                     'finetune_{}.tar'.format(str(total_iters))])
                torch.save({'iters': total_iters,
                            'state_dict': hdsm_model.state_dict(),
                            'train_loss': total_train_loss / len(train_img_loader)},
                           save_filename)

                if input_args.persist_to_s3:
                    persist_saved_models(input_args.experiment_name, input_args.savemodel)


        log.scalar_summary('train/loss', total_train_loss / len(train_img_loader), epoch)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
