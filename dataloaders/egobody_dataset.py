from typing import Dict
from yacs.config import CfgNode
from os.path import basename
import pickle as pkl
import smplx
import pandas as pd
from torch.utils import data

from .augmentation import get_example
from utils.other_utils import *
from utils.geometry import *


class DatasetEgobody(data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 dataset_file: str,
                 data_root: str,
                 train: bool = True,
                 split='train',
                 spacing=1,
                 add_scale=1.0,
                 device=None,
                 do_augment=False,
                 scene_type='whole_scene',
                 scene_cano=False,
                 scene_downsample_rate=1,
                 get_diffuse_feature=False,
                 body_rep_stats_dir='',
                 load_stage1_transl=False,
                 stage1_result_path='',
                 scene_crop_by_stage1_transl=False,
                 ):
        """
        Dataset class used for loading images and corresponding annotations.
        """
        super(DatasetEgobody, self).__init__()
        self.train = train
        self.split = split
        self.cfg = cfg
        self.device = device
        self.do_augment = do_augment

        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        self.fx_norm_coeff = self.cfg.CAM.FX_NORM_COEFF
        self.fy_norm_coeff = self.cfg.CAM.FY_NORM_COEFF
        self.cx_norm_coeff = self.cfg.CAM.CX_NORM_COEFF
        self.cy_norm_coeff = self.cfg.CAM.CY_NORM_COEFF

        self.data_root = data_root
        self.data = np.load(dataset_file)
        with open(os.path.join(self.data_root, 'transf_matrices_all_seqs.pkl'), 'rb') as fp:
            self.transf_matrices = pkl.load(fp)

        self.imgname = self.data['imgname']

        [self.imgname, self.seq_names, _] = zip(*[get_right_full_img_pth(x, self.data_root) for x in self.imgname])   # absolute dir
        self.seq_names = [basename(x) for x in self.seq_names][::spacing]
        self.imgname = self.imgname[::spacing]

        body_permutation_2d = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]  # for openpose 25 topology
        body_permutation_3d = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]  # for smpl 24 topology
        self.flip_2d_keypoint_permutation = body_permutation_2d
        self.flip_3d_keypoint_permutation = body_permutation_3d

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center'][::spacing]
        self.scale = self.data['scale'][::spacing] * add_scale

        self.has_smpl = np.ones(len(self.imgname))
        self.body_pose = self.data['pose'].astype(np.float)[::spacing]  # [n_sample, 69]
        self.betas = self.data['shape'].astype(np.float)[::spacing]
        self.global_orient_pv = self.data['global_orient_pv'].astype(np.float)[::spacing]  # [n_sample, 3]
        self.transl_pv = self.data['transl_pv'].astype(np.float)[::spacing]

        self.cx = self.data['cx'].astype(np.float)[::spacing]
        self.cy = self.data['cy'].astype(np.float)[::spacing]
        self.fx = self.data['fx'].astype(np.float)[::spacing]
        self.fy = self.data['fy'].astype(np.float)[::spacing]


        keypoints_openpose = self.data['valid_keypoints'][::spacing]
        self.keypoints_2d = keypoints_openpose
        self.keypoints_3d_pv = self.data['3d_joints_pv'].astype(np.float)[::spacing]

        # Get gender data, if available
        gender = self.data['gender'][::spacing]
        self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)

        self.load_stage1_transl = load_stage1_transl
        if self.load_stage1_transl:
            with open(stage1_result_path, 'rb') as fp:
                stage1_result = pkl.load(fp)
            self.stage1_transl_full = stage1_result['pred_cam_full_list'].astype(np.float)[::spacing]  # [n_samples, 3]

        ######## get mean/var for body representation feature in EgoHMR(to normalize for diffusion model)
        if get_diffuse_feature and split == 'train' and self.train:
            # 144-d
            global_orient_pv_all = torch.from_numpy(self.global_orient_pv).float()
            body_pose_all = torch.from_numpy(self.body_pose).float()
            full_pose_aa_all = torch.cat([global_orient_pv_all, body_pose_all], dim=1).reshape(-1, 24, 3)  # [n, 24, 3]
            full_pose_rotmat_all = aa_to_rotmat(full_pose_aa_all.reshape(-1, 3)).view(-1, 24, 3, 3)  # [bs, 24, 3, 3]
            full_pose_rot6d_all = rotmat_to_rot6d(full_pose_rotmat_all.reshape(-1, 3, 3),
                                                  rot6d_mode='diffusion').reshape(-1, 24, 6).reshape(-1, 24 * 6)  # [n, 144]
            full_pose_rot6d_all = full_pose_rot6d_all.detach().cpu().numpy()
            Xmean = full_pose_rot6d_all.mean(axis=0)  # [d]
            Xstd = full_pose_rot6d_all.std(axis=0)  # [d]
            stats_root = os.path.join(body_rep_stats_dir, 'preprocess_stats')
            os.makedirs(stats_root) if not os.path.exists(stats_root) else None
            Xstd[0:6] = Xstd[0:6].mean() / 1.0  # for global orientation
            Xstd[6:] = Xstd[6:].mean() / 1.0  # for body pose
            np.savez_compressed(os.path.join(stats_root, 'preprocess_stats.npz'), Xmean=Xmean, Xstd=Xstd)
            print('[INFO] mean/std for body_rep saved.')


        self.smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male')
        self.smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female')

        self.dataset_len = len(self.imgname)
        print('[INFO] find {} samples in {}.'.format(self.dataset_len, dataset_file))

        ########### read scene pcd
        self.scene_type = scene_type
        # self.scene_cube_normalize = scene_cube_normalize
        if self.scene_type == 'whole_scene':
            with open(os.path.join(self.data_root, 'Egohmr_scene_preprocess_s1_release/pcd_verts_dict_{}.pkl'.format(split)), 'rb') as f:
                self.pcd_verts_dict_whole_scene = pkl.load(f)
            with open(os.path.join(self.data_root, 'Egohmr_scene_preprocess_s1_release/map_dict_{}.pkl'.format(split)), 'rb') as f:
                self.pcd_map_dict_whole_scene = pkl.load(f)
        elif self.scene_type == 'cube':
            if not scene_crop_by_stage1_transl:
                self.pcd_root = os.path.join(self.data_root, 'Egohmr_scene_preprocess_cube_s2_from_gt_release')
            else:
                self.pcd_root = os.path.join(self.data_root, 'Egohmr_scene_preprocess_cube_s2_from_pred_release')
        else:
            print('[ERROR] wrong scene_type!')
            exit()


        df = pd.read_csv(os.path.join(self.data_root, 'data_info_release.csv'))
        recording_name_list = list(df['recording_name'])
        scene_name_list = list(df['scene_name'])
        self.scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        self.add_trans = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.scene_cano = scene_cano
        self.scene_downsample_rate = scene_downsample_rate



    def get_transf_matrices_per_frame(self, img_name, seq_name):
        transf_mtx_seq = self.transf_matrices[seq_name]
        kinect2holo = transf_mtx_seq['trans_kinect2holo'].astype(np.float32)  # [4,4], one matrix for all frames in the sequence
        holo2pv_dict = transf_mtx_seq['trans_world2pv']  # a dict, # frames items, each frame is a 4x4 matrix
        timestamp = basename(img_name).split('_')[0]
        holo2pv = holo2pv_dict[str(timestamp)].astype(np.float32)
        return kinect2holo, holo2pv



    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        image_file = os.path.join(self.data_root, self.imgname[idx])  # absolute path
        seq_name = self.seq_names[idx]
        keypoints_2d = self.keypoints_2d[idx].copy()  # [25, 3], openpose joints
        keypoints_3d = self.keypoints_3d_pv[idx][0:24].copy()  # [24, 3], smpl joints

        center = self.center[idx].copy().astype(np.float32)
        center_x = center[0]
        center_y = center[1]
        bbox_size = self.scale[idx].astype(np.float32) * 200
        body_pose = self.body_pose[idx].copy().astype(np.float32)  # 69
        betas = self.betas[idx].copy().astype(np.float32)  # [10]
        global_orient = self.global_orient_pv[idx].copy().astype(np.float32)  # 3
        transl = self.transl_pv[idx].copy().astype(np.float32)  # 3
        gender = self.gender[idx].copy()

        fx = self.fx[idx].copy()
        fy = self.fy[idx].copy()
        cx = self.cx[idx].copy()
        cy = self.cy[idx].copy()

        smpl_params = {'global_orient': global_orient,
                       'transl': transl,
                       'body_pose': body_pose,
                       'betas': betas
                      }
        has_smpl_params = {'global_orient': True,
                           'transl': True,
                           'body_pose': True,
                           'betas': True
                           }
        smpl_params_is_axis_angle = {'global_orient': True,
                                     'transl': False,
                                     'body_pose': True,
                                     'betas': False
                                    }

        item = {}
        item['transf_kinect2holo'], item['transf_holo2pv'] = self.get_transf_matrices_per_frame(image_file, seq_name)

        pcd_trans_kinect2pv = np.matmul(item['transf_holo2pv'], item['transf_kinect2holo'])
        pcd_trans_kinect2pv = np.matmul(self.add_trans, pcd_trans_kinect2pv)
        temp = "/".join(image_file.split('/')[-5:])
        if self.scene_type == 'whole_scene':
            scene_pcd_verts = self.pcd_verts_dict_whole_scene[self.pcd_map_dict_whole_scene[temp]]  # [20000, 3], in kinect main coord
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_kinect2pv)
        elif self.scene_type == 'cube':
            recording_name = image_file.split('/')[-4]
            img_name = image_file.split('/')[-1]
            scene_pcd_path = os.path.join(self.pcd_root, self.split, recording_name, image_file.split('/')[-3], img_name[:-3]+'npy')
            scene_pcd_verts = np.load(scene_pcd_path)  # in scene coord
            # transformation from master kinect RGB camera to scene mesh
            calib_trans_dir = os.path.join(self.data_root, 'calibrations', recording_name)
            cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')
            with open(os.path.join(cam2world_dir, self.scene_name_dict[recording_name] + '.json'), 'r') as f:
                trans_scene_to_main = np.array(json.load(f)['trans'])
            trans_scene_to_main = np.linalg.inv(trans_scene_to_main)
            pcd_trans_scene2pv = np.matmul(pcd_trans_kinect2pv, trans_scene_to_main)
            scene_pcd_verts = points_coord_trans(scene_pcd_verts, pcd_trans_scene2pv)  # nowall: 5000, withwall: 5000+30*30*5=9500

        #################################### data augmentation
        augm_config = self.cfg.DATASETS.CONFIG
        img_patch, keypoints_2d_crop_auge, keypoints_2d_vis_mask, keypoints_2d_full_auge, \
            scene_pcd_verts_full_auge, keypoints_3d_crop_auge, keypoints_3d_full_auge, smpl_params, has_smpl_params, \
            center_x_auge, center_y, cam_cx_auge, auge_scale, rotated_img \
            = get_example(image_file, center_x, center_y, bbox_size, bbox_size,
                          keypoints_2d, keypoints_3d, smpl_params, has_smpl_params,
                          self.flip_2d_keypoint_permutation, self.flip_3d_keypoint_permutation,
                          self.img_size, self.img_size, self.mean, self.std,
                          self.do_augment, augm_config,
                          fx, cam_cx=cx, cam_cy=cy,
                          scene_pcd_verts=scene_pcd_verts,
                          smpl_male=self.smpl_male, smpl_female=self.smpl_female, gender=gender)

        item['img'] = img_patch
        item['imgname'] = image_file
        item['orig_img'] = rotated_img  # original img rotate around (center_x_auge, center_y_auge)
        ###### 2d joints
        item['keypoints_2d'] = keypoints_2d_crop_auge.astype(np.float32)  # [25, 3]
        item['orig_keypoints_2d'] = keypoints_2d_full_auge.astype(np.float32)
        item['keypoints_2d_vis_mask'] = keypoints_2d_vis_mask  # [25] vis mask for openpose joint in augmented cropped img

        ###### 3d joints
        item['keypoints_3d'] = keypoints_3d_crop_auge.astype(np.float32)  # [24, 3]
        item['keypoints_3d_full'] = keypoints_3d_full_auge.astype(np.float32)

        ###### smpl params
        item['smpl_params'] = smpl_params
        for key in item['smpl_params'].keys():
            item['smpl_params'][key] = item['smpl_params'][key].astype(np.float32)
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        # item['idx'] = idx
        item['gender'] = gender

        ###### camera params
        item['fx'] = (fx / self.fx_norm_coeff).astype(np.float32)
        item['fy'] = (fy / self.fy_norm_coeff).astype(np.float32)
        item['cam_cx'] = cam_cx_auge.astype(np.float32)
        item['cam_cy'] = cy.astype(np.float32)
        ###### bbox params
        item['box_center'] = np.array([center_x_auge, center_y]).astype(np.float32)
        item['box_size'] = (bbox_size * auge_scale).astype(np.float32)

        ###### scene verts
        scene_pcd_verts_full_auge = scene_pcd_verts_full_auge.astype(np.float32)  # [n_pts, 3]
        scene_pcd_verts_full_auge = scene_pcd_verts_full_auge[::self.scene_downsample_rate]
        item['scene_pcd_verts_full'] = scene_pcd_verts_full_auge  # [20000, 3]
        # only for test
        if self.load_stage1_transl:
            item['stage1_transl_full'] = self.stage1_transl_full[idx].astype(np.float32)

        return item
