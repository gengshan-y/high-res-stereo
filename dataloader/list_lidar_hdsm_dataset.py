import os

def dataloader(filepath):
    train_file = f'{filepath}/train.txt'
    with open(train_file) as f:
        lines = [l.strip() for l in f.readlines()]
    samples = [os.path.join(filepath, l) for l in lines]
    im0 = [f'{sample}/im0.png' for sample in samples]
    im1 = [f'{sample}/im1.png' for sample in samples]
    disp0 = [f'{sample}/disp0GT.pfm' for sample in samples]
    disp1 = [f'{sample}/disp1GT.pfm' for sample in samples]
    return im0, im1, disp0, disp1
