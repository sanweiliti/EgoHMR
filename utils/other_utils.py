import torch
import logging
import datetime
import os, json
import numpy as np


def get_logger(logdir):
    logger = logging.getLogger('emotion')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths


def get_transf_matrices_per_frame(transf_matrices, img_name, seq_name):
    transf_mtx_seq = transf_matrices[seq_name]
    kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
    holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix

    timestamp = os.path.basename(img_name).split('_')[0]
    holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
    return kinect2holo, holo2pv


def parse_img_full_path(img_full_path):
    path_splitted = img_full_path.split('/')
    img_basename = path_splitted[-1]  # '132754997786014666_frame_01442.jpg'
    session = path_splitted[-5]
    seq = path_splitted[-4]
    fpv_recording_name = path_splitted[-3]  # '2021-09-07-164904'

    return session, seq, fpv_recording_name, img_basename


def get_right_full_img_pth(imgname_in_npz, img_dir):
    session, seq, fpv_recording_name, img_basename = parse_img_full_path(imgname_in_npz)
    img_pth_final = os.path.join(img_dir, imgname_in_npz)
    record_folder_path = os.path.join(img_dir, session, seq)
    return img_pth_final, record_folder_path, fpv_recording_name


def recursive_to(x, target):
    """
    Recursively transfer a batch of data to the target device
    Args:
        x (Any): Batch of data.
        target (torch.device): Target device.
    Returns:
        Batch of data where all tensors are transfered to the target device.
    """
    if isinstance(x, dict):
        return {k: recursive_to(v, target) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(target)
    elif isinstance(x, list):
        return [recursive_to(i, target) for i in x]
    else:
        return x


SMPL_EDGES = [(0, 1),
              [0, 2],
              [0, 3],
              [1, 4],
              [2, 5],
              [3, 6],
              [4, 7],
              [5, 8],
              [6, 9],
              [7, 10],
              [8, 11],
              [9, 12],
              [9, 13],
              [9, 14],
              [12, 15],
              [13, 16],
              [14, 17],
              [16, 18],
              [17, 19],
              [18, 20],
              [19, 21],
              [20, 22],
              [21, 23]]