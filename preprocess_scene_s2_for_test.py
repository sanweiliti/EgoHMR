"""
Preprocess script for EgoBody for second stage of EgoHMR (i.e., the scene-conditioned diffusion model for local body pose).
Scene vertices around the predicted body (from the first stage of EgoHMR) in a 2x2m cube will be cropped and saved.
The preprocessed scene vertices are used for test and evaluation of EgoHMR.
"""
import open3d as o3d
import numpy as np
import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import pyrender
import smplx
import torch
import copy
import random
import math
import argparse
from utils.other_utils import *
from utils.geometry import *

parser = argparse.ArgumentParser(description='ProHMR training code')
parser.add_argument('--stage1_result_path', type=str, default='output_results_release/output_53618/results.pkl', help='')
parser.add_argument('--scene_verts_num_target', type=int, default=20000, help='numbers of scene vertices to crop')
parser.add_argument('--cube_size', type=int, default=2, help='cropped scene cube size')
parser.add_argument('--split', type=str, default='val', help='val/train/test')
parser.add_argument('--data_root', type=str, default='/mnt/ssd/egobody_release/', help='path to egobody data')
parser.add_argument('--save_root', type=str, default='/mnt/ssd/egobody_release/Egohmr_scene_preprocess_cube_s2_from_pred',
                    help='path to save preprocessed scene point cloud')

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transf_matrices_per_frame(transf_matrices, img_name, seq_name):
    transf_mtx_seq = transf_matrices[seq_name]
    kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
    holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix

    timestamp = os.path.basename(img_name).split('_')[0]
    holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
    return kinect2holo, holo2pv


if __name__ == '__main__':
    with open(args.stage1_result_path, 'rb') as fp:
        stage1_result = pkl.load(fp)
    stage1_transl_full_all = stage1_result['pred_cam_full_list']  # [n, 3]

    df = pd.read_csv(os.path.join(args.data_root, 'data_info_release.csv'))
    recording_name_list = list(df['recording_name'])
    scene_name_list = list(df['scene_name'])
    body_idx_fpv_list = list(df['body_idx_fpv'])
    body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
    scene_name_dict = dict(zip(recording_name_list, scene_name_list))

    data = np.load(os.path.join(args.data_root, 'smpl_spin_npz/egocapture_{}_smpl.npz'.format(args.split)))
    with open(os.path.join(args.data_root, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
        transf_matrices = pkl.load(fp)

    imgname_list = data['imgname']  # 'egocentric_color/...'
    betas_list = data['shape']
    global_orient_pv_list = data['global_orient_pv']
    transl_pv_list = data['transl_pv']
    body_pose_list = data['pose']
    gender_list = data['gender']

    fx_list = data['fx']
    fy_list = data['fy']
    cam_cx_list = data['cx']
    cam_cy_list = data['cy']

    [imgname_list, seqname_list, _] = zip(*[get_right_full_img_pth(x, args.data_root) for x in imgname_list])   # absolute dir
    seqname_list = [os.path.basename(x) for x in seqname_list]

    body_model_male = smplx.create('data/smpl', model_type='smpl', gender='male').to(device)
    body_model_female = smplx.create('data/smpl', model_type='smpl', gender='female').to(device)

    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

    add_trans = np.array([[1.0, 0, 0, 0],
                          [0, -1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 1]])

    orig_scene_mesh_dict = {}

    ######################## visualize 3d bodies and scene
    step = 1
    cnt = 0
    map_dict = {}
    pcd_verts_cube_nowall_dict = {}
    pcd_verts_cube_withwall_dict = {}
    last_scene_name = ''
    for i in tqdm(range(0, len(imgname_list), step)):
        imgname = imgname_list[i]
        recording_name = imgname.split('/')[-4]
        holo_recording_time = imgname.split('/')[-3]
        frame_name = imgname.split('/')[-1][0:-4]
        frame_id = imgname.split('/')[-1][-15:-4]
        scene_name = scene_name_dict[recording_name]

        if cnt % 1 == 0 or last_scene_name != scene_name:
            trans_kinect2holo, trans_holo2pv = get_transf_matrices_per_frame(transf_matrices, imgname_list[i], seqname_list[i])

            ######## read scene
            if scene_name not in orig_scene_mesh_dict.keys():
                scene_dir = os.path.join(os.path.join(args.data_root, 'scene_mesh'), '{}/{}.obj'.format(scene_name, scene_name))
                orig_scene_mesh_dict[scene_name] = o3d.io.read_triangle_mesh(scene_dir, print_progress=True)
                orig_scene_mesh_dict[scene_name].compute_vertex_normals()

            calib_trans_dir = os.path.join(args.data_root, 'calibrations', recording_name)
            cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master kinect RGB camera to scene mesh
            with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
                trans_scene_to_main = np.array(json.load(f)['trans'])
            trans_scene_to_main = np.linalg.inv(trans_scene_to_main)

            ###### get stage 1 transl in current pv coord
            cur_stage1_transl_pv = stage1_transl_full_all[[i]]  # [1, 3]?
            trans_scene2pv = np.matmul(trans_kinect2holo, trans_scene_to_main)
            trans_scene2pv = np.matmul(trans_holo2pv, trans_scene2pv)
            trans_scene2pv = np.matmul(add_trans, trans_scene2pv)
            cur_stage1_transl_scene = points_coord_trans(cur_stage1_transl_pv, np.linalg.inv(trans_scene2pv))[0]

            cur_scene_mesh = copy.deepcopy(orig_scene_mesh_dict[scene_name])

            ##################### visualization, for debugging
            # ###### get gt body in current pv coord
            # torch_param = {}
            # torch_param['transl'] = torch.tensor(transl_pv_list[[i]]).float().to(device)
            # torch_param['global_orient'] = torch.tensor(global_orient_pv_list[[i]]).float().to(device)
            # torch_param['betas'] = torch.tensor(betas_list[[i]]).float().to(device)
            # torch_param['body_pose'] = torch.tensor(body_pose_list[[i]]).float().to(device)
            # if gender_list[i] == 'm':
            #     output = body_model_male(return_verts=True, **torch_param)
            # elif gender_list[i] == 'f':
            #     output = body_model_female(return_verts=True, **torch_param)
            # body_verts = output.vertices.detach().cpu().numpy().squeeze()  # [6890, 3]
            # body_joints = output.joints.detach().cpu().numpy().squeeze()
            #
            # gt_body_o3d_pv = o3d.geometry.TriangleMesh()
            # gt_body_o3d_pv.vertices = o3d.utility.Vector3dVector(body_verts)  # [6890, 3]
            # gt_body_o3d_pv.triangles = o3d.utility.Vector3iVector(body_model_male.faces)
            # gt_body_o3d_pv.compute_vertex_normals()
            # gt_body_o3d_scene = copy.deepcopy(gt_body_o3d_pv)
            # gt_body_o3d_scene.transform(np.linalg.inv(add_trans))
            # gt_body_o3d_scene.transform(np.linalg.inv(trans_holo2pv))
            # gt_body_o3d_scene.transform(np.linalg.inv(trans_kinect2holo))
            # gt_body_o3d_scene.transform(np.linalg.inv(trans_scene_to_main))
            # body_verts_scene = np.asarray(gt_body_o3d_scene.vertices)  # [6890, 3]
            #
            # cur_scene_mesh.transform(trans_scene_to_main)
            # cur_scene_mesh.transform(trans_kinect2holo)
            # cur_scene_mesh.transform(trans_holo2pv)
            # cur_scene_mesh.transform(add_trans)
            #
            # transformation = np.identity(4)
            # transformation[:3, 3] = cur_stage1_transl_scene
            # stage1_transl_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            # stage1_transl_sphere.paint_uniform_color([70 / 255, 130 / 255, 180 / 255])  # steel blue 70,130,180
            # stage1_transl_sphere.compute_vertex_normals()
            # stage1_transl_sphere.transform(transformation)
            #
            # o3d.visualization.draw_geometries([cur_scene_mesh, mesh_frame, stage1_transl_sphere])
            # o3d.visualization.draw_geometries([cur_scene_mesh, mesh_frame, gt_body_o3d_scene])
            # o3d.visualization.draw_geometries([cur_scene_mesh, mesh_frame, gt_body_o3d_scene, stage1_transl_sphere])



            ###### cut/dowmsample scene
            ######### note! in scene coord, y axis up
            scene_verts = np.asarray(cur_scene_mesh.vertices)  # [n, 3]

            # get body center as the predicted body translation from egohmr 1st stage
            # body_center = np.mean(body_verts_scene, axis=0)  # [x, y, z]
            body_center = cur_stage1_transl_scene
            # rotate scene (pv coord) around body center
            rot_angle = random.uniform(0, 2 * (math.pi))
            scene_verts_aug = np.zeros(scene_verts.shape)
            scene_verts_aug[:, 0] = (scene_verts[:, 0] - body_center[0]) * math.cos(rot_angle) - (scene_verts[:, 2] - body_center[2]) * math.sin(rot_angle) + body_center[0]
            scene_verts_aug[:, 2] = (scene_verts[:, 0] - body_center[0]) * math.sin(rot_angle) + (scene_verts[:, 2] - body_center[2]) * math.cos(rot_angle) + body_center[2]
            scene_verts_aug[:, 1] = scene_verts[:, 1]

            # cropped scene verts in the cube
            min_x = body_center[0] - args.cube_size / 2
            max_x = body_center[0] + args.cube_size / 2
            min_y = body_center[2] - args.cube_size / 2
            max_y = body_center[2] + args.cube_size / 2
            scene_verts_auge_crop = scene_verts_aug[np.where((scene_verts_aug[:, 0] >= min_x) & (scene_verts_aug[:, 0] <= max_x) &
                                                             (scene_verts_aug[:, 2] >= min_y) & (scene_verts_aug[:, 2] <= max_y))]
            scene_verts_auge_crop = scene_verts_auge_crop[scene_verts_auge_crop[:, 1] <= np.min(scene_verts_auge_crop[:, 1]) + args.cube_size]

            ##### downsample the points
            scene_pcd_auge_crop = o3d.geometry.PointCloud()
            scene_pcd_auge_crop.points = o3d.utility.Vector3dVector(scene_verts_auge_crop)
            n_verts= len(scene_verts_auge_crop)
            if n_verts < args.scene_verts_num_target:
                print('[ERROR]', 'scene vertex number', n_verts, '<', 'scene_verts_num_target', args.scene_verts_num_target)
                exit()
            downsample_rate = int(n_verts / args.scene_verts_num_target)
            scene_pcd_auge_crop = scene_pcd_auge_crop.uniform_down_sample(every_k_points=downsample_rate)
            scene_verts_auge_crop_downsample = np.asarray(scene_pcd_auge_crop.points)[0:args.scene_verts_num_target]

            ####################### reansform back to original scene space ##########################
            scene_verts_crop_downsample = np.zeros(scene_verts_auge_crop_downsample.shape)
            scene_verts_crop_downsample[:, 0] = (scene_verts_auge_crop_downsample[:, 0] - body_center[0]) * math.cos(-rot_angle) - (scene_verts_auge_crop_downsample[:, 2] - body_center[2]) * math.sin(-rot_angle) + body_center[0]
            scene_verts_crop_downsample[:, 2] = (scene_verts_auge_crop_downsample[:, 0] - body_center[0]) * math.sin(-rot_angle) + (scene_verts_auge_crop_downsample[:, 2] - body_center[2]) * math.cos(-rot_angle) + body_center[2]
            scene_verts_crop_downsample[:, 1] = scene_verts_auge_crop_downsample[:, 1]


            # ####### visualize
            # scene_pcd_crop_nowall_downsample = o3d.geometry.PointCloud()
            # scene_pcd_crop_nowall_downsample.points = o3d.utility.Vector3dVector(scene_verts_crop_nowall_downsample)
            # o3d.visualization.draw_geometries([scene_pcd_crop_nowall_downsample, mesh_frame, stage1_transl_sphere, gt_body_o3d_scene])

            ######## save pcds in scene coord system
            # /mnt/ssd/proHMR_scene_preprocess/recording_name/2021-09-07-155421/timestamp_frame_xxxxx.obj
            if not os.path.exists(os.path.join(args.save_root, args.split, recording_name, holo_recording_time)):
                os.makedirs(os.path.join(args.save_root, args.split, recording_name, holo_recording_time))
            save_path = os.path.join(args.save_root, args.split, recording_name, holo_recording_time, frame_name+'.npy')
            np.save(save_path, scene_verts_crop_downsample)

        # # save a dict: map each pv frame name to the nearest saved scene pcd frame name: for dataloader later
        cnt += 1
        last_scene_name = scene_name

    print('Completed scene point clouds preprocessing, results saved to {}.'.format(args.save_root))

