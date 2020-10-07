import os

from dataloader.listfiles import airsim_dataloader, lidar_dataloader


def test_airsim_dataloader(_sync_data):
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/airsim')
    lefts, rights, disps = airsim_dataloader(dataset_path)
    assert len(lefts) == 15
    assert len(rights) == 15
    assert len(disps) == 15
    for (left, right, disp) in zip(lefts, rights, disps):
        left_folder = os.path.dirname(left)
        assert left_folder == os.path.dirname(right)
        assert left_folder == os.path.dirname(disp)
        assert 'left_bgr.png' in os.path.basename(left)
        assert 'right_bgr.png' in os.path.basename(right)
        assert 'left_disparity.npy' in os.path.basename(disp)


def test_lidar_dataloader(_sync_data):
    dataset_path = os.path.join(os.path.dirname(__file__), '../../data/lidar')
    lefts, rights, disps_l, disps_r = lidar_dataloader(dataset_path)
    assert len(lefts) == 2
    assert len(rights) == 2
    assert len(disps_l) == 2
    assert len(disps_r) == 2
    for (left, right, disp_l, disp_r) in zip(lefts, rights, disps_l, disps_r):
        left_folder = os.path.dirname(left)
        assert left_folder == os.path.dirname(right)
        assert left_folder == os.path.dirname(disp_l)
        assert left_folder == os.path.dirname(disp_r)
        assert 'im0.png' in os.path.basename(left)
        assert 'im1.png' in os.path.basename(right)
        assert 'disp0GT.pfm' in os.path.basename(disp_l)
        assert 'disp1GT.pfm' in os.path.basename(disp_r)
