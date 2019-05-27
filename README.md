# Hierarchical Deep Stereo Matching on High Resolution Images
<img src="data-mbtest/CrusadeP/im0.png" width="200">
<img src="mboutput/CrusadeP-disp.png" width="200">
<img src="mboutput/CrusadeP-ent.png" width="200">
left image, disparity map, and uncertainty map (brighter->higher uncertainty)

[Project website](http://www.contrib.andrew.cmu.edu/~gengshay/cvpr19stereo)

## Requirements
- python 2.7.15
- pytorch 0.4.0

## Weights
[Download](https://drive.google.com/file/d/1BlH7IafX-X0A5kFPd50WkZXqxo0_gtoI/view?usp=sharing)

## Data
[High-res-real-stereo (HR-RS)](https://drive.google.com/file/d/1UTkOgw5IO-GcVYapzCdzrmjbjkGMyOH4/view?usp=sharing)

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

## Parameters
- testres: 1 is full resolution, and 0.5 is half resolution, and so on
- max_disp: maximum disparity range to search
- clean: threshold of cleaning. clean=0 means removing all the pixels.

## Citation
```
@InProceedings{Yang2019HSM,
  author    = {Gengshan Yang and Joshua Manela and Michael Happold and Deva Ramanan},
  title     = {Hierarchical Deep Stereo Matching on High-resolution Images},
  booktitle = CVPR,
  year      = {2019},
}
```

## Acknowledgement
Part of the code is borrowed from [PSMNet](https://github.com/JiaRenChang/PSMNet) and [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

