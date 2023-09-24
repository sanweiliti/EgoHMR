"""
Preprocess script for EgoBody for first stage of EgoHMR (i.e., ProHMR-scene baseline).
Scene vertices in front of the egocentric camera will be cropped and saved.
Vertices behind of the camera are ignored.
"""

import open3d as o3d
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import copy
import argparse
from utils.other_utils import *



parser = argparse.ArgumentParser(description='ProHMR training code')
parser.add_argument('--scene_verts_num_target', type=int, default=20000, help='numbers of scene vertices to crop')
parser.add_argument('--split', type=str, default='train', help='val/train/test split of egobody')
parser.add_argument('--data_root', type=str, default='/mnt/ssd/egobody_release/', help='path to egobody data')
parser.add_argument('--save_root', type=str, default='/mnt/ssd/egobody_release/Egohmr_scene_preprocess_s1',
                    help='path to save preprocessed scene point cloud')
args = parser.parse_args()



if __name__ == '__main__':
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root, exist_ok=True)

    ################################ read dataset information ################################
    df = pd.read_csv(os.path.join(args.data_root, 'data_info_release.csv'))
    recording_name_list = list(df['recording_name'])
    scene_name_list = list(df['scene_name'])
    scene_name_dict = dict(zip(recording_name_list, scene_name_list))

    data = np.load(os.path.join(args.data_root, 'smpl_spin_npz/egocapture_{}_smpl.npz'.format(args.split)))
    with open(os.path.join(args.data_root, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        transf_matrices = pkl.load(fp)

    imgname_list = data['imgname']  # 'egocentric_color/...'
    fx_list = data['fx']
    fy_list = data['fy']
    cam_cx_list = data['cx']
    cam_cy_list = data['cy']

    [imgname_list, seqname_list, _] = zip(*[get_right_full_img_pth(x, args.data_root) for x in imgname_list])  # absolute dir
    seqname_list = [os.path.basename(x) for x in seqname_list]

    # additional transformation from opengl coord  to opencv system: ego camera - opengl coord, kinect camera - opencv coord
    add_trans = np.array([[1.0, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])
    orig_scene_mesh_dict = {}
    step = 1
    cnt = 0
    map_dict = {}
    pcd_verts_dict = {}
    last_scene_name = ''
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    ################################## scene pcd processing ##################################
    for i in tqdm(range(0, len(imgname_list), step)):
        imgname = imgname_list[i]
        recording_name = imgname.split('/')[-4]
        holo_recording_time = imgname.split('/')[-3]
        frame_name = imgname.split('/')[-1][0:-4]
        frame_id = imgname.split('/')[-1][-15:-4]
        scene_name = scene_name_dict[recording_name]

        # cnt: only save processed scene point clouds every 15 frames
        # reason 1: to save disk space
        # reason 2: the scene point clouds do not change much within a small time window
        if cnt % 15 == 0 or last_scene_name != scene_name:
            trans_kinect2holo, trans_holo2pv = get_transf_matrices_per_frame(transf_matrices, imgname_list[i], seqname_list[i])

            ########################  read scene mesh  ########################
            if scene_name not in orig_scene_mesh_dict.keys():
                scene_dir = os.path.join(os.path.join(args.data_root, 'scene_mesh'), '{}/{}.obj'.format(scene_name, scene_name))
                orig_scene_mesh_dict[scene_name] = o3d.io.read_triangle_mesh(scene_dir, enable_post_processing=True, print_progress=True)
                orig_scene_mesh_dict[scene_name].compute_vertex_normals()

            calib_trans_dir = os.path.join(args.data_root, 'calibrations', recording_name)
            cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master kinect RGB camera to scene mesh
            with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
                trans_scene_to_main = np.array(json.load(f)['trans'])
            trans_scene_to_main = np.linalg.inv(trans_scene_to_main)

            ######################## coord transform ########################
            # scene mesh space --> master kinect space --> pv cam space
            cur_scene_mesh = copy.deepcopy(orig_scene_mesh_dict[scene_name])
            cur_scene_mesh.transform(trans_scene_to_main)
            cur_scene_mesh.transform(trans_kinect2holo)
            cur_scene_mesh.transform(trans_holo2pv)
            cur_scene_mesh.transform(add_trans)

            ####################### crop scene mesh ########################
            # to keep only scene verts in front of the egocentric camera
            scene_verts = np.asarray(cur_scene_mesh.vertices)  # [n, 3]
            mask = scene_verts[:, -1] > 0  # in egocentric camear coord, discard verts in front of the camera
            scene_verts = scene_verts[mask]

            ####################### dowmsample scene verts ########################
            # to a certain vertex number
            scene_pcd = o3d.geometry.PointCloud()
            scene_pcd.points = o3d.utility.Vector3dVector(scene_verts)
            n_verts = len(scene_verts)
            downsample_rate = int(n_verts / args.scene_verts_num_target)
            scene_pcd = scene_pcd.uniform_down_sample(every_k_points=downsample_rate)
            # scene pcd in current pv cam coordinate frame system
            scene_verts = np.asarray(scene_pcd.points)  # [n, 3]
            scene_pcd.points = o3d.utility.Vector3dVector(scene_verts[0:args.scene_verts_num_target])

            # transform back to original kinect master coordinate frame
            scene_pcd.transform(np.linalg.inv(add_trans))
            scene_pcd.transform(np.linalg.inv(trans_holo2pv))
            scene_pcd.transform(np.linalg.inv(trans_kinect2holo))
            # o3d.visualization.draw_geometries([scene_pcd, mesh_frame])  # in master kinect coord frame

            imgname_temp_save_pcd = "/".join(imgname.split('/')[4:])
            pcd_verts_dict[imgname_temp_save_pcd] = np.asarray(scene_pcd.points)


        # map each ego frame name to the nearest saved scene pcd frame name, for dataloader later
        cnt += 1
        last_scene_name = scene_name
        imgname_temp = "/".join(imgname.split('/')[4:])
        map_dict[imgname_temp] = imgname_temp_save_pcd


    ######################### save pcd verts in npy ########################
    # saved scene pcd in master kinect coord frame
    with open(os.path.join(args.save_root, 'map_dict_{}.pkl'.format(args.split)), 'wb') as result_file:
        pkl.dump(map_dict, result_file, protocol=2)
    with open(os.path.join(args.save_root, 'pcd_verts_dict_{}.pkl'.format(args.split)), 'wb') as result_file:
        pkl.dump(pcd_verts_dict, result_file, protocol=2)


