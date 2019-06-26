<img src="http://www.contrib.andrew.cmu.edu/~gengshay/wordpress/wp-content/uploads/2019/06/cvpr19-middlebury1.gif" width="400">
qualitative results on Middlebury
<img src="./middlebury-benchmark.png" width="400">
performance on Middlebury benchmark (y-axis: the lower the better)
<img src="data-mbtest/CrusadeP/im0.png" width="400">
left image
<img src="mboutput/CrusadeP/capture_000.png" width="400">
3D projection
<img src="mboutput/CrusadeP-disp.png" width="400">
disparity map
<img src="mboutput/CrusadeP-ent.png" width="400">
uncertainty map (brighter->higher uncertainty)
# Hierarchical Deep Stereo Matching on High Resolution Images

## Requirements
- python 2.7.15
- pytorch 0.4.0

## Weights
[Download](https://drive.google.com/file/d/1BlH7IafX-X0A5kFPd50WkZXqxo0_gtoI/view?usp=sharing)

## Data

### train/val
- [Middlebury (train set and additional images)](https://drive.google.com/file/d/1jJVmGKTDElyKiTXoj6puiK4vUY9Ahya7/view?usp=sharing)
- [High-res-virtual-stereo (HR-VS)](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view?usp=sharing)
- [KITTI-2012&2015](http://www.cvlibs.net/datasets/kitti/eval_stereo.php)
- [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)

### test
[High-res-real-stereo (HR-RS)](): comming soon

## Train
1. Download and extract training data in folder /d/. Training data include Middlebury train set, HR-VS, KITTI-12/15 and SceneFlow.
2. Run
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --maxdisp 384 --batchsize 24 --database /d/ --logname log1 --savemodel /somewhere/  --epochs 10
```
3. Evalute on Middlebury additional images and KITTI validation set. After 10 epochs, average error on Middlebury *additional* images with half-res should be around 4.6 (excluding Shopvac).

## Inference
Example:
```
CUDA_VISIBLE_DEVICES=3 python submission.py --datapath ./data-mbtest/   --outdir ./mboutput --loadmodel ./weights/final-768px.tar  --testres 1 --clean 0.8 --max_disp -1
```

Evaluation:
```
CUDA_VISIBLE_DEVICES=3 python submission.py --datapath ./data-HRRS/   --outdir ./output --loadmodel ./weights/final-768px.tar  --testres 0.5
python eval_disp.py --indir ./output --gtdir ./data-HRRS/
```

And use [cvkit](https://github.com/roboception/cvkit) to visualize in 3D.

## Parameters
- testres: 1 is full resolution, and 0.5 is half resolution, and so on
- max_disp: maximum disparity range to search
- clean: threshold of cleaning. clean=0 means removing all the pixels.

## Citation
```
@InProceedings{yang2019hsm,
author = {Yang, Gengshan and Manela, Joshua and Happold, Michael and Ramanan, Deva},
title = {Hierarchical Deep Stereo Matching on High-Resolution Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Acknowledgement
Part of the code is borrowed from [MiddEval-SDK](http://vision.middlebury.edu/stereo/submit3/), [PSMNet](https://github.com/JiaRenChang/PSMNet), [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch) and [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).
