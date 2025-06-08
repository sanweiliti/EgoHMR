import argparse
from tqdm import tqdm
from configs import get_config, prohmr_config
import smplx
import pandas as pd
import pickle as pkl
import random
from utils.pytorch3d_chamfer_distance import chamfer_distance


from utils.pose_utils import *
from utils.renderer import *
from utils.other_utils import *
from utils.geometry import *
from dataloaders.egobody_dataset import DatasetEgobody
from utils.geometry import perspective_projection

from models.egohmr.egohmr_volsmpl import EgoHMRVolsmpl
from diffusion.model_util import create_gaussian_diffusion



parser = argparse.ArgumentParser(description='Evaluate trained models')
parser.add_argument('--two_stage', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='if use the predicted body/cam translation from stage 1 (prohmr-scene)')
parser.add_argument('--scene_crop_by_stage1_transl', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='for the scene input, if use the ones cropped by the predicted body transl from stage 1 (prohmr-scene)')
parser.add_argument('--stage1_result_path', type=str, default='output_results_release/output_prohmr_scene_53618/results.pkl',
                    help='path to stage 1 (prohmr-scene) result path')

parser.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')
parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoints_egohmr/91453/best_model_mpjpe_vis.pt', help='path to trained checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (configs/prohmr.yaml)')
parser.add_argument('--batch_size', type=int, default=5, help='Batch size for inference')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=100, help='How often to print results')
parser.add_argument("--seed", default=0, type=int)

####### save/render/vis args
parser.add_argument('--render', default='False', type=lambda x: x.lower() in ['true', '1'], help='render pred body mesh on images')
parser.add_argument('--render_multi_sample', default='True', type=lambda x: x.lower() in ['true', '1'], help='render all pred samples for input image')
parser.add_argument('--output_render_root', default='output_render', help='output folder to save rendered images')
parser.add_argument('--render_step', type=int, default=80, help='how often to render results')

parser.add_argument('--vis_o3d', default='False', type=lambda x: x.lower() in ['true', '1'], help='visualize 3d body and scene with open3d')
parser.add_argument('--vis_o3d_gt', default='False', type=lambda x: x.lower() in ['true', '1'], help='if visualize ground truth body as well')
parser.add_argument('--vis_step', type=int, default=8, help='how often to visualize 3d results')  # 8/1

parser.add_argument('--save_results', default='False', type=lambda x: x.lower() in ['true', '1'], help='save sampled results')
parser.add_argument('--save_root', type=str, default='output_results', help='output folder to save pred camera/body transl')

#### scene args
parser.add_argument('--scene_cano', default='True', type=lambda x: x.lower() in ['true', '1'], help='transl scene points to be human-centric')
parser.add_argument('--scene_type', type=str, default='cube', choices=['whole_scene', 'cube'],
                    help='whole_scene (all scene vertices in front of camera) / cube (a 2x2 cube around the body)')

#### diffusion model args
parser.add_argument("--num_diffusion_timesteps", default=50, type=int, help='total steps for diffusion')
parser.add_argument('--timestep_respacing_eval', type=str, default='ddpm', choices=['ddim5', 'ddpm'], help='ddim/ddpm sampling schedule')
parser.add_argument('--diffuse_fuse', default='True', type=lambda x: x.lower() in ['true', '1'], help='if to use classifier-free sampling')
parser.add_argument('--with_volsmpl_grad', default='True', type=lambda x: x.lower() in ['true', '1'], help='if use collision-guided sampling')
parser.add_argument('--cond_grad_weight', type=float, default=30.0, help='weight for collision gradient')
parser.add_argument('--only_mask_img_cond', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='only mask img features during trainig with cond_mask_prob')
parser.add_argument('--pelvis_vis_loosen', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='set pelvis joint visibility the same as knees, allows more flexibility for lower body diversity')

#### eval args
parser.add_argument("--eval_spacing", default=1, type=int, help='downsample test set by #')
parser.add_argument('--num_samples', type=int, default=1, help='Number of test samples to draw')
parser.add_argument('--eval_coll_loss', default='True', type=lambda x: x.lower() in ['true', '1'], help='if evaluate human-scene collision score')
parser.add_argument('--eval_contact_score', default='True', type=lambda x: x.lower() in ['true', '1'], help='if evaluate human-scene contact score')
parser.add_argument('--eval_with_vis_mask_pa', default='True', type=lambda x: x.lower() in ['true', '1'], help='if calculate pa-mpjpe by pa-alignment from visible joints')


parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true focal length as input')
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true camera center as input')
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'], help='take bbox info as input')
parser.add_argument('--add_bbox_scale', type=float, default=1.2, help='scale orig bbox size')
parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'], help='shuffle in dataloader')

args = parser.parse_args()

def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
fixseed(args.seed)

# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test():
    ############################## Load model config
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    # Update number of test samples drawn to the desired value
    model_cfg.defrost()
    model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
    model_cfg.freeze()


    ############################### Setup and load pretrained model, diffusion
    logdir = '/'.join(args.checkpoint.split('/')[0:-1])
    preprocess_stats = np.load(os.path.join(logdir, 'preprocess_stats/preprocess_stats.npz'))
    body_rep_mean = torch.from_numpy(preprocess_stats['Xmean']).float().to(device)
    body_rep_std = torch.from_numpy(preprocess_stats['Xstd']).float().to(device)
    model = EgoHMRVolsmpl(cfg=model_cfg, device=device,
                   body_rep_mean=body_rep_mean, body_rep_std=body_rep_std,
                   with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info,
                   with_cam_center=args.with_cam_center,
                   scene_feat_dim=512, scene_type=args.scene_type, scene_cano=args.scene_cano,
                   cond_mask_prob=0.0, only_mask_img_cond=args.only_mask_img_cond,
                   pelvis_vis_loosen=args.pelvis_vis_loosen, diffuse_fuse=args.diffuse_fuse)

    if args.timestep_respacing_eval == 'ddpm':
        args.timestep_respacing_eval = ''
    diffusion_sample = create_gaussian_diffusion(num_diffusion_timesteps=args.num_diffusion_timesteps, timestep_respacing=args.timestep_respacing_eval,
                                          body_rep_mean=body_rep_mean, body_rep_std=body_rep_std)

    weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(weights['state_dict'], strict=False)
    model.eval()
    print('load traind mode from:', args.checkpoint)
    print('diffusion sample method:', args.timestep_respacing_eval)

    ################################ Create dataset and data loader
    test_dataset = DatasetEgobody(cfg=model_cfg, train=False, device=device, data_root=args.dataset_root,
                                  dataset_file=os.path.join(args.dataset_root, 'annotation_egocentric_smpl_npz/egocapture_test_smpl.npz'),
                                  spacing=args.eval_spacing,
                                  add_scale=args.add_bbox_scale, split='test',
                                  scene_type=args.scene_type, scene_cano=args.scene_cano,
                                  load_stage1_transl=args.two_stage,
                                  stage1_result_path=args.stage1_result_path,
                                  scene_crop_by_stage1_transl=args.scene_crop_by_stage1_transl)
    dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    ################################# load smpl models
    smpl_neutral = smplx.create('data/smpl', model_type='smpl', gender='neutral', create_transl=False, batch_size=args.batch_size).to(device)
    smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male', batch_size=args.batch_size).to(device)
    smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female', batch_size=args.batch_size).to(device)


    ################################# create dir to save rendered results
    # cannot render and visualize in 3d at the same time
    if args.render and args.vis_o3d:
        print('[ERROR]: cannot set both args.render and args.vis_o3d to True!')
        exit()
    if args.render:
        model_id = args.checkpoint.split('/')[-2]
        output_vis_folder = os.path.join(args.output_render_root, 'output_egohmr_{}'.format(model_id))
        os.makedirs(output_vis_folder) if not os.path.exists(output_vis_folder) else None
        #### setup renderer
        camera_pose = np.eye(4)
        camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
        material_pred = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(1.0, 193 / 255, 193 / 255, 1.0)
        )
        renderer = pyrender.OffscreenRenderer(viewport_width=1920, viewport_height=1080)
        #### setup camera view for rendering in scene (use the master kinect pose in egobody)
        camera_params_dir = os.path.join(args.dataset_root, 'kinect_cam_params')  # intrinsics
        with open(os.path.join(camera_params_dir, 'kinect_master', 'Color.json'), 'r') as f:
            color_cam = json.load(f)
        [f_x_kinect, f_y_kinect] = color_cam['f']
        [c_x_kinect, c_y_kinect] = color_cam['c']
        df = pd.read_csv(os.path.join(args.dataset_root, 'data_info_release.csv'))
        recording_name_list = list(df['recording_name'])
        scene_name_list = list(df['scene_name'])
        scene_name_dict = dict(zip(recording_name_list, scene_name_list))
        scane_mesh_dict = {}
        recording_name_dict = {}

    ################################# read scene name for each sequence for 3d visualization
    if args.vis_o3d:
        import open3d as o3d
        df = pd.read_csv(os.path.join(args.dataset_root, 'data_info_release.csv'))
        recording_name_list = list(df['recording_name'])
        scene_name_list = list(df['scene_name'])
        scene_name_dict = dict(zip(recording_name_list, scene_name_list))

    if args.render or args.vis_o3d:
        add_trans = np.array([[1.0, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, -1, 0],
                              [0, 0, 0, 1]])

    ################################# create list for evaluation metrics
    # accuracy
    g_mpjpe_all = np.zeros([len(test_dataset), args.num_samples])
    mpjpe_all = np.zeros([len(test_dataset), args.num_samples])
    pa_mpjpe_all = np.zeros([len(test_dataset), args.num_samples])
    v2v_all = np.zeros([len(test_dataset), args.num_samples])  # translation/pelv-aligned
    # diversity
    std_joints_all = np.zeros([len(test_dataset)])
    apd_joints_all = np.zeros([len(test_dataset)])

    g_mpjpe_vis_all_list = np.zeros([len(test_dataset), args.num_samples])
    g_mpjpe_invis_all_list = np.zeros([len(test_dataset), args.num_samples])
    mpjpe_vis_all_list = np.zeros([len(test_dataset), args.num_samples])
    mpjpe_invis_all_list = np.zeros([len(test_dataset), args.num_samples])
    pa_mpjpe_vis_all_list = np.zeros([len(test_dataset), args.num_samples])
    pa_mpjpe_invis_all_list = np.zeros([len(test_dataset), args.num_samples])
    v2v_vis_all_list = np.zeros([len(test_dataset), args.num_samples])
    v2v_invis_all_list = np.zeros([len(test_dataset), args.num_samples])
    joint_vis_num_list, joint_invis_num_list = [], []
    vertex_vis_num_list, vertex_invis_num_list = [], []
    # diversity
    std_joints_vis_all = np.zeros([len(test_dataset)])
    std_joints_invis_all = np.zeros([len(test_dataset)])
    apd_joints_vis_all = np.zeros([len(test_dataset)])
    apd_joints_invis_all = np.zeros([len(test_dataset)])
    # scene plausibility
    contact_ratio_list_all = np.zeros([len(test_dataset), args.num_samples])
    coll_ratio_list_all = np.zeros([len(test_dataset), args.num_samples])


    pred_betas_list = []
    pred_body_pose_list = []
    pred_global_orient_list = []
    pred_cam_full_list = []
    gt_cam_full_list = []

    ################################# test
    for step, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        curr_batch_size = batch['img'].shape[0]
        bbox_size = batch['box_size'].to(device)  # [bs]
        bbox_center = batch['box_center'].to(device)  # [bs, 2]
        focal_length = batch['fx'] * model_cfg.CAM.FX_NORM_COEFF  # [bs]
        cam_cx = batch['cam_cx']
        cam_cy = batch['cam_cy']
        gt_cam_full = batch['smpl_params']['transl'].to(device).clone()

        with torch.no_grad():
            ######## load predicted translation from stage 1
            if args.two_stage:
                batch['smpl_params']['transl'] = batch['stage1_transl_full']  # replace gt camera by pred camera
                pred_cam_full = batch['stage1_transl_full']

            ######## iterate for multiple samples
            shape = [curr_batch_size, 144]
            out_all_samples = {}
            out_all_samples['pred_smpl_params'] = {}
            for n in range(args.num_samples):
                out_cur_sample = diffusion_sample.val_losses(model=model, batch=batch, shape=shape, progress=False,
                                                             clip_denoised=False, cur_epoch=0,
                                                             timestep_respacing=args.timestep_respacing_eval,
                                                             cond_fn_with_grad=args.with_volsmpl_grad, cond_grad_weight=args.cond_grad_weight)
                if args.eval_coll_loss:
                    ######## calculate human-scene collision metric
                    coll_ratio_list = model.eval_coll_volsmpl(out_cur_sample)  # [bs]
                    coll_ratio_list = np.array(coll_ratio_list)
                    coll_ratio_list_all[step * args.batch_size:step * args.batch_size + curr_batch_size, n] = coll_ratio_list
                for key in out_cur_sample['pred_smpl_params'].keys():
                    if key not in out_all_samples['pred_smpl_params'].keys():
                        out_all_samples['pred_smpl_params'][key] = []
                    out_all_samples['pred_smpl_params'][key].append(out_cur_sample['pred_smpl_params'][key].unsqueeze(1))
            for key in out_cur_sample['pred_smpl_params'].keys():
                out_all_samples['pred_smpl_params'][key] = torch.cat(out_all_samples['pred_smpl_params'][key], dim=1)  # [bs, n_sample, ...]

        ###### get gt annotations
        gt_pose = {}
        gt_pose['global_orient'] = batch['smpl_params']['global_orient'].to(device)
        gt_pose['transl'] = gt_cam_full
        gt_pose['body_pose'] = batch['smpl_params']['body_pose'].to(device)
        gt_pose['betas'] = batch['smpl_params']['betas'].to(device)
        gender = batch['gender'].to(device)

        if curr_batch_size != args.batch_size:
            smpl_neutral = smplx.create('data/smpl', model_type='smpl', gender='neutral', create_transl=False, batch_size=curr_batch_size).to(device)
            smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male', batch_size=curr_batch_size).to(device)
            smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female', batch_size=curr_batch_size).to(device)

        ###### get pred smpl params
        pred_betas = out_all_samples['pred_smpl_params']['betas']  # [bs, n_sample, 10]
        pred_body_pose = out_all_samples['pred_smpl_params']['body_pose']  # [bs, n_sample, 23, 3, 3]
        pred_global_orient = out_all_samples['pred_smpl_params']['global_orient']  # [bs, n_sample, 1, 3, 3]

        pred_betas_list.append(pred_betas)
        pred_body_pose_list.append(pred_body_pose)
        pred_global_orient_list.append(pred_global_orient)

        ###### get pred smpl joints / vertices
        pred_output = smpl_neutral(betas=pred_betas.reshape(-1, 10), body_pose=pred_body_pose.reshape(-1, 23, 3, 3),
                                   global_orient=pred_global_orient.reshape(-1, 1, 3, 3), pose2rot=False)
        pred_vertices = pred_output.vertices.reshape(curr_batch_size, -1, 6890, 3)  # [bs, n_sample, 6890, 3]
        pred_keypoints_3d = pred_output.joints.reshape(curr_batch_size, -1, 45, 3)[:, :, 0:24, :]  # [bs, n_sample, 24, 3]
        pred_pelvis = pred_keypoints_3d[:, :, [0], :].clone()  # [bs, n_sample, 1, 3]
        pred_keypoints_3d_align = pred_keypoints_3d - pred_pelvis
        pred_vertices_align = pred_vertices - pred_pelvis

        ##### get pred cam joints / vertices in full img coord
        pred_vertices_full = pred_vertices + batch['smpl_params']['transl'].unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 6890, 3]
        pred_keypoints_3d_full = pred_keypoints_3d + batch['smpl_params']['transl'].unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 24, 3]
        if args.two_stage:
            pred_cam_full_list.append(pred_cam_full)
        gt_cam_full_list.append(gt_cam_full)

        ##### get gt body joints / vertices
        gt_body = smpl_male(**gt_pose)
        gt_joints = gt_body.joints
        gt_vertices = gt_body.vertices
        gt_body_female = smpl_female(**gt_pose)
        gt_joints_female = gt_body_female.joints
        gt_vertices_female = gt_body_female.vertices
        gt_joints[gender == 1, :, :] = gt_joints_female[gender == 1, :, :]
        gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]

        gt_keypoints_3d = gt_joints[:, :24, :]  # [bs, 24, 3]
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()  # [bs,1,3]
        gt_keypoints_3d_align = gt_keypoints_3d - gt_pelvis
        gt_vertices_align = gt_vertices - gt_pelvis

        ######################## visualize 3d bodies and scene in the original physical camera
        if args.vis_o3d and step % args.vis_step == 0:
            # for bi in range(curr_batch_size):  # todo: uncomment if want to visualize for all images within this batch
            for bi in range(1):
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                pred_body_o3d_full = o3d.geometry.TriangleMesh()
                pred_body_o3d_full.vertices = o3d.utility.Vector3dVector(pred_vertices_full[bi, 0].detach().cpu().numpy())
                pred_body_o3d_full.triangles = o3d.utility.Vector3iVector(smpl_neutral.faces)
                pred_body_o3d_full.compute_vertex_normals()

                if args.vis_o3d_gt:
                    gt_body_o3d_full = o3d.geometry.TriangleMesh()
                    gt_body_o3d_full.vertices = o3d.utility.Vector3dVector(gt_vertices[bi].detach().cpu().numpy())  # [6890, 3]
                    gt_body_o3d_full.triangles = o3d.utility.Vector3iVector(smpl_neutral.faces)
                    gt_body_o3d_full.compute_vertex_normals()
                    gt_body_o3d_full.paint_uniform_color([0.0, 0.0, 1.0])

                ################ read scene point cloud from data loader
                cur_scene_pcd = batch['scene_pcd_verts_full'][bi]
                cur_scene_pcd = cur_scene_pcd.detach().cpu().numpy()
                scene_pcd = o3d.geometry.PointCloud()
                scene_pcd.points = o3d.utility.Vector3dVector(cur_scene_pcd)

                ################ read scene
                recording_name = batch['imgname'][bi].split('/')[-4]
                scene_name = scene_name_dict[recording_name]
                scene_dir = os.path.join(os.path.join(args.dataset_root, 'scene_mesh'), '{}/{}.obj'.format(scene_name, scene_name))
                scene_mesh_full = o3d.io.read_triangle_mesh(scene_dir, enable_post_processing=True, print_progress=True)
                scene_mesh_full.compute_vertex_normals()

                # transform scene from scene coord to current pv coord
                calib_trans_dir = os.path.join(args.dataset_root, 'calibrations', recording_name)
                cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master kinect RGB camera to scene mesh
                with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
                    trans_scene_to_main = np.array(json.load(f)['trans'])
                trans_scene_to_main = np.linalg.inv(trans_scene_to_main)
                T_kinect2holoseq = batch['transf_kinect2holo'].to(device)
                T_holoseq2holoframe = batch['transf_holo2pv'].to(device)
                trans_kinect2holo = T_kinect2holoseq[bi].cpu().numpy()
                cur_world2pv_transform = T_holoseq2holoframe[bi].cpu().numpy()

                scene_mesh_full.transform(trans_scene_to_main)
                scene_mesh_full.transform(trans_kinect2holo)
                scene_mesh_full.transform(cur_world2pv_transform)
                scene_mesh_full.transform(add_trans)

                ################ visualize
                if args.vis_o3d_gt:
                    o3d.visualization.draw_geometries([scene_pcd, scene_mesh_full, mesh_frame, pred_body_o3d_full, gt_body_o3d_full])
                else:
                    o3d.visualization.draw_geometries([scene_pcd, scene_mesh_full, mesh_frame, pred_body_o3d_full])

        ################################# calculate evaluation metrics
        ###### project gt_keypoints_3d on img plane to get visibility mask
        focal_length_proj = focal_length.unsqueeze(-1).repeat(1, 2)  # [bs, 2]
        camera_center_full = torch.cat([cam_cx.unsqueeze(-1), cam_cy.unsqueeze(-1)], dim=-1)  # [bs, 2]
        gt_keypoints_2d_full = perspective_projection(gt_keypoints_3d,
                                                      translation=torch.zeros([curr_batch_size, 3]).to(device),
                                                      camera_center=camera_center_full,
                                                      focal_length=focal_length_proj)  # [bs, 24, 2]
        gt_vertices_2d_full = perspective_projection(gt_vertices,
                                                     translation=torch.zeros([curr_batch_size, 3]).to(device),
                                                     camera_center=camera_center_full,
                                                     focal_length=focal_length_proj)  # [bs, 6890, 2]

        joint_vis_mask = (gt_keypoints_2d_full[:, :, 0] >= 0) * (gt_keypoints_2d_full[:, :, 0] < 1920) * \
                         (gt_keypoints_2d_full[:, :, 1] >= 0) * (gt_keypoints_2d_full[:, :, 1] < 1080)  # [bs, 24]
        vertex_vis_mask = (gt_vertices_2d_full[:, :, 0] >= 0) * (gt_vertices_2d_full[:, :, 0] < 1920) * \
                          (gt_vertices_2d_full[:, :, 1] >= 0) * (gt_vertices_2d_full[:, :, 1] < 1080)  # [bs, 24]

        # visible joint/vertex num for each image
        joint_vis_num = joint_vis_mask.sum().item()
        joint_vis_num_list.append(joint_vis_num)  # [n_all_frames]
        joint_invis_num_list.append(curr_batch_size * gt_keypoints_3d.shape[-2] - joint_vis_num)
        vertex_vis_num = vertex_vis_mask.sum().item()
        vertex_vis_num_list.append(vertex_vis_num)  # [n_all_frames]
        vertex_invis_num_list.append(curr_batch_size * gt_vertices.shape[-2] - vertex_vis_num)

        # G-MPJPE
        gmpjpe_per_joint = torch.sqrt(((pred_keypoints_3d_full - gt_keypoints_3d.unsqueeze(1)) ** 2).sum(dim=-1))  # [bs, n_sample, 24]
        gmpjpe_cur_batch = gmpjpe_per_joint.mean(dim=-1).cpu().numpy()  # [bs, n_sample]
        g_mpjpe_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = gmpjpe_cur_batch
        # g-mpjpe_vis/invis
        gmpjpe_cur_batch_vis = (gmpjpe_per_joint * joint_vis_mask.unsqueeze(1)).sum(dim=-1).cpu().numpy()  # [bs, n_sample]
        gmpjpe_cur_batch_invis = (gmpjpe_per_joint * (~(joint_vis_mask.unsqueeze(1)))).sum(dim=-1).cpu().numpy()
        g_mpjpe_vis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = gmpjpe_cur_batch_vis
        g_mpjpe_invis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = gmpjpe_cur_batch_invis

        # MPJPE
        mpjpe_per_joint = torch.sqrt(((pred_keypoints_3d_align - gt_keypoints_3d_align.unsqueeze(1)) ** 2).sum(dim=-1))  # [bs, n_sample, 24]
        mpjpe_cur_batch = mpjpe_per_joint.mean(dim=-1).cpu().numpy()  # [bs, n_sample]
        mpjpe_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = mpjpe_cur_batch
        # mpjpe_vis/invis
        mpjpe_cur_batch_vis = (mpjpe_per_joint * joint_vis_mask.unsqueeze(1)).sum(dim=-1).cpu().numpy()  # [bs, n_sample]
        mpjpe_cur_batch_invis = (mpjpe_per_joint * (~(joint_vis_mask.unsqueeze(1)))).sum(dim=-1).cpu().numpy()
        mpjpe_vis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = mpjpe_cur_batch_vis
        mpjpe_invis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = mpjpe_cur_batch_invis

        # PA-MPJPE
        n_joints = gt_keypoints_3d_align.shape[-2]
        if not args.eval_with_vis_mask_pa:
            pampjpe_per_joint = reconstruction_error(
                pred_keypoints_3d_align.reshape(-1, n_joints, 3).cpu().numpy(),
                gt_keypoints_3d_align.unsqueeze(1).repeat(1, args.num_samples, 1, 1).reshape(-1, n_joints, 3).cpu().numpy(),
                avg_joint=False).reshape(curr_batch_size, args.num_samples, -1)  # [bs, num_samples, 24]
        else:
            pampjpe_per_joint = reconstruction_error_with_vis_mask(
                joint_vis_mask.unsqueeze(1).unsqueeze(-1).repeat(1, args.num_samples, 1, 3).reshape(-1, n_joints, 3).cpu().numpy(),
                pred_keypoints_3d_align.reshape(-1, n_joints, 3).cpu().numpy(),
                gt_keypoints_3d_align.unsqueeze(1).repeat(1, args.num_samples, 1, 1).reshape(-1, n_joints, 3).cpu().numpy(),
                avg_joint=False).reshape(curr_batch_size, args.num_samples, -1)  # [bs, num_samples, 24]
        pampjpe_cur_batch = pampjpe_per_joint.mean(axis=-1)  # [bs, num_samples]
        pa_mpjpe_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = pampjpe_cur_batch
        # pampjpe_vis/invis
        pampjpe_cur_batch_vis = (pampjpe_per_joint * joint_vis_mask.unsqueeze(1).cpu().numpy()).sum(axis=-1)
        pampjpe_cur_batch_invis = (pampjpe_per_joint * (~(joint_vis_mask.unsqueeze(1)).cpu().numpy())).sum(axis=-1)
        pa_mpjpe_vis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = pampjpe_cur_batch_vis
        pa_mpjpe_invis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = pampjpe_cur_batch_invis

        # V2V
        v2v_per_verts = torch.sqrt(((pred_vertices_align - gt_vertices_align.unsqueeze(1)) ** 2).sum(dim=-1))  # [bs, n_sample, 6890]
        v2v_cur_batch = v2v_per_verts.mean(dim=-1).cpu().numpy()
        v2v_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = v2v_cur_batch
        # v2v-vis/invis
        v2v_cur_batch_vis = (v2v_per_verts * vertex_vis_mask.unsqueeze(1)).sum(dim=-1).cpu().numpy()  # [bs, n_sample]
        v2v_cur_batch_invis = (v2v_per_verts * (~(vertex_vis_mask.unsqueeze(1)))).sum(dim=-1).cpu().numpy()
        v2v_vis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = v2v_cur_batch_vis
        v2v_invis_all_list[step * args.batch_size:step * args.batch_size + curr_batch_size] = v2v_cur_batch_invis

        ############ diversity std joints
        # pred_keypoints_3d_align  # [bs, n_samples, 24, 3]
        std_joints_cur_batch = torch.std(pred_keypoints_3d_align, dim=1, unbiased=True).mean(dim=-1).mean(dim=-1).cpu().numpy()  # [bs]
        std_joints_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = std_joints_cur_batch

        #### diversity vis/in-vis
        std_joints_vis_cur_batch = []
        for k in range(curr_batch_size):
            temp = pred_keypoints_3d_align[k, :, joint_vis_mask[k]]  # [num_samples, num_vis_joints, 3]
            std_joints_vis_cur_sample = torch.std(temp, dim=0, unbiased=True).mean(-1).mean(-1).item()
            std_joints_vis_cur_batch.append(std_joints_vis_cur_sample)
        std_joints_vis_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = std_joints_vis_cur_batch

        std_joints_invis_cur_batch = []
        for k in range(curr_batch_size):
            temp = pred_keypoints_3d_align[k, :, ~joint_vis_mask[k]]  # [num_samples, num_vis_joints, 3]  could be nan if all joint are visible
            std_joints_invis_cur_sample = torch.std(temp, dim=0, unbiased=True).mean(-1).mean(-1).item()
            std_joints_invis_cur_batch.append(std_joints_invis_cur_sample)
        std_joints_invis_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = std_joints_invis_cur_batch


        ############### diversity apd joints
        # pred_keypoints_3d_align  # [bs, n_samples, 24, 3]
        a = pred_keypoints_3d_align.detach().cpu().numpy()
        n_samples = a.shape[1]
        pairwise_dist = np.linalg.norm(a[:, None, :, :, :] - a[:, :, None, :, :], axis=-1)  # [bs, n_samples, n_samples, 24]
        apd_joints = pairwise_dist.sum(axis=(-1, -2, -3)) / a.shape[-2] / n_samples / (n_samples - 1) / 2  # [bs]
        apd_joints_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = apd_joints

        #### diversity vis/in-vis
        apd_joints_vis_cur_batch = []
        for k in range(curr_batch_size):
            temp = pred_keypoints_3d_align[k, :, joint_vis_mask[k]].detach().cpu().numpy()  # [num_samples, num_vis_joints, 3]
            pairwise_dist_vis_cur_sample = np.linalg.norm(temp[None, :, :, :] - temp[:, None, :, :], axis=-1)  # [n_samples, n_samples, num_vis_joints]
            apd_joints_cur_sample = pairwise_dist_vis_cur_sample.sum() / temp.shape[-2] / n_samples / (n_samples - 1) / 2
            apd_joints_vis_cur_batch.append(apd_joints_cur_sample)
        apd_joints_vis_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = apd_joints_vis_cur_batch

        apd_joints_invis_cur_batch = []
        for k in range(curr_batch_size):
            temp = pred_keypoints_3d_align[k, :, ~joint_vis_mask[k]].detach().cpu().numpy()  # [num_samples, num_vis_joints, 3]
            pairwise_dist_invis_cur_sample = np.linalg.norm(temp[None, :, :, :] - temp[:, None, :, :], axis=-1)  # [n_samples, n_samples, num_vis_joints]
            apd_joints_cur_sample = pairwise_dist_invis_cur_sample.sum() / temp.shape[-2] / n_samples / (n_samples - 1) / 2
            apd_joints_invis_cur_batch.append(apd_joints_cur_sample)
        apd_joints_invis_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = apd_joints_invis_cur_batch

        ###### contact: chamfer distance
        if args.eval_contact_score:
            pred_vertices_full_copy = pred_vertices_full.clone().reshape(-1, 6890, 3)  # [bs*n_samples, 6890, 3]
            cur_scene_pcd = batch['scene_pcd_verts_full']  # [bs, 20000, 3]
            cur_scene_pcd = cur_scene_pcd.unsqueeze(1).repeat(1, args.num_samples, 1, 1).reshape(-1, cur_scene_pcd.shape[1], 3)  # [bs*n_samples, 20000, 3]
            contact_dist, _, _ = chamfer_distance(pred_vertices_full_copy.contiguous(),
                                                  cur_scene_pcd.contiguous())  # [bs*n_samples, 6890]
            # if minimum dist from body to scene is < thres, count as in contact
            contact_cur_batch = (contact_dist.min(dim=-1)[0] < 0.02).reshape(curr_batch_size, -1)  # [bs, n_samples]
            contact_ratio_list_all[step * args.batch_size:step * args.batch_size + curr_batch_size] = contact_cur_batch.cpu().numpy()

        if step % args.log_freq == 0 and step > 0:
            ######## compute from N random samples
            print('--------- mode: compute mean from {} random samples ---------'.format(args.num_samples))
            error_dict = {
                'G-MPJPE': 1000 * g_mpjpe_all[:step * args.batch_size].mean(),
                'G-MPJPE-vis': 1000 * g_mpjpe_vis_all_list[:step * args.batch_size].sum() / sum(joint_vis_num_list) / args.num_samples,
                'G-MPJPE-invis': 1000 * g_mpjpe_invis_all_list[:step * args.batch_size].sum() / sum(joint_invis_num_list) / args.num_samples,
                'MPJPE': 1000 * mpjpe_all[:step * args.batch_size].mean(),
                'MPJPE-vis': 1000 * mpjpe_vis_all_list[:step * args.batch_size].sum() / sum(joint_vis_num_list) / args.num_samples,
                'MPJPE-invis': 1000 * mpjpe_invis_all_list[:step * args.batch_size].sum() / sum(joint_invis_num_list) / args.num_samples,
                'PA-MPJPE': 1000 * pa_mpjpe_all[:step * args.batch_size, ].mean(),
                'PA-MPJPE-vis': 1000 * pa_mpjpe_vis_all_list[:step * args.batch_size].sum() / sum(joint_vis_num_list) / args.num_samples,
                'PA-MPJPE-invis': 1000 * pa_mpjpe_invis_all_list[:step * args.batch_size].sum() / sum(joint_invis_num_list) / args.num_samples,
                'V2V': 1000 * v2v_all[:step * args.batch_size].mean(),
                'V2V-vis': 1000 * v2v_vis_all_list[:step * args.batch_size].sum() / sum(vertex_vis_num_list) / args.num_samples,
                'V2V-invis': 1000 * v2v_invis_all_list[:step * args.batch_size].sum() / sum(vertex_invis_num_list) / args.num_samples,
            }
            plausible_dict = {
                'collision-ratio': coll_ratio_list_all[:step * args.batch_size].mean() if args.eval_coll_loss else -1,
                'contact-ratio': contact_ratio_list_all[:step * args.batch_size].mean() if args.eval_contact_score else -1,
            }
            print('G-MPJPE all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['G-MPJPE'], error_dict['G-MPJPE-vis'], error_dict['G-MPJPE-invis']))
            print('MPJPE all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['MPJPE'], error_dict['MPJPE-vis'], error_dict['MPJPE-invis']))
            print('PA-MPJPE all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['PA-MPJPE'], error_dict['PA-MPJPE-vis'], error_dict['PA-MPJPE-invis']))
            print('V2V all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['V2V'], error_dict['V2V-vis'], error_dict['V2V-invis']))
            print('collision / contact ratio: {:0.5f} / {:0.2f}'.format(plausible_dict['collision-ratio'], plausible_dict['contact-ratio']))

            ######## compute with smallest-error-sample from N samples
            print('--------- mode: compute with smallest-error-sample-invis from {} samples ---------'.format(args.num_samples))
            select_idx = np.argmin(mpjpe_invis_all_list[:step * args.batch_size], axis=1)
            print('MPJPE-invis: ' + str(1000 * mpjpe_invis_all_list[range(step * args.batch_size), select_idx].sum() / sum(joint_invis_num_list)))

            ####### mean std over all data/all samples/all joints/xyz coords
            print('--------- diversity ---------')
            diversity_dict = {
                'std-joints': 1000 * std_joints_all[:step * args.batch_size].mean(),
                'std-joints-vis': 1000 * std_joints_vis_all[:step * args.batch_size].mean(),
                'std-joints-invis': 1000 * (std_joints_invis_all[:step * args.batch_size]
                                            [~np.isnan(std_joints_invis_all[:step * args.batch_size])].mean()),
                'apd-joints': 1000 * apd_joints_all[:step * args.batch_size].mean(),
                'apd-joints-vis': 1000 * apd_joints_vis_all[:step * args.batch_size].mean(),
                'apd-joints-invis': 1000 * (apd_joints_invis_all[:step * args.batch_size]
                                            [~np.isnan(apd_joints_invis_all[:step * args.batch_size])].mean()),
            }
            print('std-joints all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(diversity_dict['std-joints'], diversity_dict['std-joints-vis'], diversity_dict['std-joints-invis']))
            print('apd-joints all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(diversity_dict['apd-joints'], diversity_dict['apd-joints-vis'], diversity_dict['apd-joints-invis']))

        ########################## visualize
        ##### vis_multi_sample=True: render for all samples, otherwise only render one sample
        if args.render and step % args.render_step == 0:
            if args.render_multi_sample:
                vis_sample_n = args.num_samples
            else:
                vis_sample_n = 1

            # for bi in range(curr_batch_size):  # todo: uncomment if want to render for all images within this batch
            for bi in range(1):
                cur_img_path = batch['imgname'][bi]
                full_img_input = cv2.imread(batch['imgname'][bi])  # [h, w, 3]
                start_point = [int((bbox_center[bi][0] - bbox_size[bi] / 2).item()),
                               int((bbox_center[bi][1] - bbox_size[bi] / 2).item())]
                end_point = [int((bbox_center[bi][0] + bbox_size[bi] / 2).item()),
                             int((bbox_center[bi][1] + bbox_size[bi] / 2).item())]
                full_img_input = cv2.rectangle(img=full_img_input, pt1=(start_point[0], start_point[1]),
                                               pt2=(end_point[0], end_point[1]), color=(0, 255, 0), thickness=3)
                recording_name = cur_img_path.split('/')[-4]
                frame_name = cur_img_path.split('/')[-1]

                output_img_list = []
                for k in range(0, vis_sample_n, 1):
                    if args.with_focal_length:
                        cur_focal_length = (batch['fx'][bi] * model_cfg.CAM.FX_NORM_COEFF).item()
                    else:
                        cur_focal_length = model_cfg.EXTRA.FOCAL_LENGTH
                    cam_cx = int(batch['cam_cx'][bi])
                    cam_cy = int(batch['cam_cy'][bi])

                    ############# render pred body overlay on input image
                    body_trimesh_pv = trimesh.Trimesh(pred_vertices_full[bi, k].detach().cpu().numpy(), smpl_neutral.faces, process=False)
                    pred_render_on_img = render_on_img(fx=cur_focal_length, fy=cur_focal_length, cx=cam_cx, cy=cam_cy,
                                                       input_img=full_img_input, body_trimesh=body_trimesh_pv,
                                                       material=material_pred, light=light, camera_pose=camera_pose,
                                                       renderer=renderer)

                    ############## render pred body in 3d scene
                    ### transform scene from scene coord to kinect coord
                    scene_name = scene_name_dict[recording_name]
                    if scene_name not in scane_mesh_dict.keys():
                        scene_dir = os.path.join(os.path.join(args.dataset_root, 'scene_mesh'), '{}/{}.obj'.format(scene_name, scene_name))
                        static_scene = trimesh.load(scene_dir)
                        scane_mesh_dict[scene_name] = static_scene
                    calib_trans_dir = os.path.join(args.dataset_root, 'calibrations', recording_name)
                    cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master kinect RGB camera to scene mesh
                    with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
                        trans_main_to_scene = np.array(json.load(f)['trans'])
                    if recording_name not in recording_name_dict.keys():
                        recording_name_dict[recording_name] = True
                        scene_trimesh_kinect = copy.deepcopy(scane_mesh_dict[scene_name])
                        scene_trimesh_kinect.apply_transform(np.linalg.inv(trans_main_to_scene))  # in master kinect RGB cam coordinate
                    ### transform body from pv cam coord to kinect coord
                    trans_kinect2holo, trans_holo2pv = batch['transf_kinect2holo'][bi].cpu().numpy(), batch['transf_holo2pv'][bi].cpu().numpy()
                    body_trimesh_master = copy.deepcopy(body_trimesh_pv)
                    body_trimesh_master.apply_transform(add_trans)
                    body_trimesh_master.apply_transform(np.linalg.inv(trans_holo2pv))
                    body_trimesh_master.apply_transform(np.linalg.inv(trans_kinect2holo))
                    ### render
                    pred_render_in_scene = render_in_scene(fx=f_x_kinect, fy=f_y_kinect, cx=c_x_kinect, cy=c_y_kinect,
                                                           body_trimesh=body_trimesh_master,
                                                           scene_trimesh=scene_trimesh_kinect,
                                                           material=material_pred, light=light, camera_pose=camera_pose,
                                                           renderer=renderer)
                    ### combine output render images
                    output_img_list.append(np.concatenate([full_img_input, pred_render_on_img, pred_render_in_scene], axis=1))

                output_img_list = np.asarray(output_img_list)
                output_img_list = output_img_list.reshape(-1, output_img_list.shape[-2], output_img_list.shape[-1])
                output_img_list = cv2.resize(src=output_img_list,
                                             dsize=((output_img_list.shape[1] // 2, output_img_list.shape[0] // 2)),
                                             interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(output_vis_folder, '{}_{}'.format(recording_name, frame_name)), output_img_list)



    print('*** Final Results ***')
    print('--------- mode: compute mean from {} random samples ---------'.format(args.num_samples))
    error_dict = {
        'G-MPJPE': 1000 * g_mpjpe_all.mean(),
        'G-MPJPE-vis': 1000 * g_mpjpe_vis_all_list.sum() / sum(joint_vis_num_list) / args.num_samples,
        'G-MPJPE-invis': 1000 * g_mpjpe_invis_all_list.sum() / sum(joint_invis_num_list) / args.num_samples,
        'MPJPE': 1000 * mpjpe_all.mean(),
        'MPJPE-vis': 1000 * mpjpe_vis_all_list.sum() / sum(joint_vis_num_list) / args.num_samples,
        'MPJPE-invis': 1000 * mpjpe_invis_all_list.sum() / sum(joint_invis_num_list) / args.num_samples,
        'PA-MPJPE': 1000 * pa_mpjpe_all.mean(),
        'PA-MPJPE-vis': 1000 * pa_mpjpe_vis_all_list.sum() / sum(joint_vis_num_list) / args.num_samples,
        'PA-MPJPE-invis': 1000 * pa_mpjpe_invis_all_list.sum() / sum(joint_invis_num_list) / args.num_samples,
        'V2V': 1000 * v2v_all.mean(),
        'V2V-vis': 1000 * v2v_vis_all_list.sum() / sum(vertex_vis_num_list) / args.num_samples,
        'V2V-invis': 1000 * v2v_invis_all_list.sum() / sum(vertex_invis_num_list) / args.num_samples,
        }
    plausible_dict = {
        'collision-ratio': coll_ratio_list_all.mean() if args.eval_coll_loss else -1,
        'contact-ratio': contact_ratio_list_all.mean() if args.eval_contact_score else -1,
    }
    print('G-MPJPE all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['G-MPJPE'], error_dict['G-MPJPE-vis'], error_dict['G-MPJPE-invis']))
    print('MPJPE all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['MPJPE'], error_dict['MPJPE-vis'], error_dict['MPJPE-invis']))
    print('PA-MPJPE all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['PA-MPJPE'], error_dict['PA-MPJPE-vis'], error_dict['PA-MPJPE-invis']))
    print('V2V all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(error_dict['V2V'], error_dict['V2V-vis'], error_dict['V2V-invis']))
    print('collision / contact ratio: {:0.5f} / {:0.2f}'.format(plausible_dict['collision-ratio'], plausible_dict['contact-ratio']))

    print('--------- mode: compute with smallest-error-sample-invis from {} samples ---------'.format(args.num_samples))
    select_idx = np.argmin(mpjpe_invis_all_list, axis=1)
    print('MPJPE-invis: ' + str(1000 * mpjpe_invis_all_list[range(len(test_dataset)), select_idx].sum() / sum(joint_invis_num_list)))

    ####### mean std over all data/all samples/all joints/xyz coords
    print('--------- diversity ---------')
    diversity_dict = {
        'std-joints': 1000 * std_joints_all.mean(),
        'std-joints-vis': 1000 * std_joints_vis_all.mean(),
        'std-joints-invis': 1000 * (std_joints_invis_all[~np.isnan(std_joints_invis_all)].mean()),
        'apd-joints': 1000 * apd_joints_all.mean(),
        'apd-joints-vis': 1000 * apd_joints_vis_all.mean(),
        'apd-joints-invis': 1000 * (apd_joints_invis_all[~np.isnan(apd_joints_invis_all)].mean()),
        }
    print('std-joints all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(diversity_dict['std-joints'], diversity_dict['std-joints-vis'], diversity_dict['std-joints-invis']))
    print('apd-joints all/vis/invis: {:0.2f} / {:0.2f} / {:0.2f}'.format(diversity_dict['apd-joints'], diversity_dict['apd-joints-vis'], diversity_dict['apd-joints-invis']))


    if args.save_results:
        pred_betas_list = torch.cat(pred_betas_list, dim=0).detach().cpu().numpy()
        pred_global_orient_list = torch.cat(pred_global_orient_list, dim=0).detach().cpu().numpy()
        pred_body_pose_list = torch.cat(pred_body_pose_list, dim=0).detach().cpu().numpy()
        if args.two_stage:
            pred_cam_full_list = torch.cat(pred_cam_full_list, dim=0).detach().cpu().numpy()
        gt_cam_full_list = torch.cat(gt_cam_full_list, dim=0).detach().cpu().numpy()

        model_id = args.checkpoint.split('/')[-2]
        output_res_folder = '{}/output_egohmr_{}'.format(args.save_root, model_id)
        os.mkdir(output_res_folder) if not os.path.exists(output_res_folder) else None
        output_res_dict = {}
        output_res_dict['pred_betas_list'] = pred_betas_list
        output_res_dict['pred_global_orient_list'] = pred_global_orient_list
        output_res_dict['pred_body_pose_list'] = pred_body_pose_list
        output_res_dict['collision_ratio_list'] = coll_ratio_list_all  # [n_data, n_samples]
        output_res_dict['contact_ratio_list'] = contact_ratio_list_all  # [n_data, n_samples]
        if args.two_stage:
            output_res_dict['pred_cam_full_list'] = pred_cam_full_list
        output_res_dict['gt_cam_full_list'] = gt_cam_full_list
        with open(os.path.join(output_res_folder, 'results_seed_{}.pkl'.format(args.seed)), 'wb') as result_file:
            pkl.dump(output_res_dict, result_file, protocol=2)
        print('[INFO] pred results saved to {}.'.format(output_res_folder))



if __name__ == '__main__':
    test()


