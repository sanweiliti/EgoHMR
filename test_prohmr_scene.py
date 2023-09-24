import argparse
from tqdm import tqdm
from configs import get_config, prohmr_config
import smplx
import pandas as pd
import pickle as pkl
import random

from dataloaders.egobody_dataset import DatasetEgobody
from models.prohmr.prohmr_scene import ProHMRScene
from utils.pose_utils import *
from utils.renderer import *
from utils.other_utils import *
from utils.geometry import *



parser = argparse.ArgumentParser(description='ProHMR-scene test code')
parser.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')
parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoints_egohmr/53618/best_model.pt', help='path to trained checkpoint')
parser.add_argument('--model_cfg', type=str, default=None, help='Path to config file. If not set use the default (configs/prohmr.yaml)')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for inference')  # 50/10
parser.add_argument('--num_samples', type=int, default=5, help='Number of test samples for each image')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers used for data loading')
parser.add_argument('--log_freq', type=int, default=100, help='How often to print evaluation results')  # 100/10
parser.add_argument("--seed", default=0, type=int)

parser.add_argument('--render', default='False', type=lambda x: x.lower() in ['true', '1'], help='render pred body mesh on images')
parser.add_argument('--render_multi_sample', default='True', type=lambda x: x.lower() in ['true', '1'], help='render all pred samples for input image')
parser.add_argument('--output_render_root', default='output_render', help='output folder to save rendered images')  #
parser.add_argument('--render_step', type=int, default=8, help='how often to render results')

parser.add_argument('--vis_o3d', default='False', type=lambda x: x.lower() in ['true', '1'], help='visualize 3d body and scene with open3d')
parser.add_argument('--vis_o3d_gt', default='False', type=lambda x: x.lower() in ['true', '1'], help='if visualize ground truth body as well')
parser.add_argument('--vis_step', type=int, default=8, help='how often to visualize 3d results')  # 8/1

parser.add_argument('--save_pred_transl', default='True', type=lambda x: x.lower() in ['true', '1'], help='save pred camera/body transl')
parser.add_argument('--save_root', type=str, default='output_results', help='output folder to save pred camera/body transl')

parser.add_argument('--scene_cano', default='False', type=lambda x: x.lower() in ['true', '1'], help='transl scene points to be human-centric')
parser.add_argument('--scene_type', type=str, default='whole_scene', choices=['whole_scene', 'cube'],
                    help='whole_scene (all scene vertices in front of camera) / cube (a 2x2 cube around the body)')

parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true focal length as input')
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true camera center as input')
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'], help='take bbox info as input')
parser.add_argument('--add_bbox_scale', type=float, default=1.2, help='scale orig bbox size')
parser.add_argument('--shuffle', default='False', type=lambda x: x.lower() in ['true', '1'], help='shuffle in dataloader')

args = parser.parse_args()


# Use the GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def fixseed(seed):
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
fixseed(args.seed)


def test():
    ############################## Load model config
    if args.model_cfg is None:
        model_cfg = prohmr_config()
    else:
        model_cfg = get_config(args.model_cfg)
    ### Update number of test samples drawn to the desired value
    model_cfg.defrost()
    model_cfg.TRAIN.NUM_TEST_SAMPLES = args.num_samples
    model_cfg.freeze()

    ############################### Setup and load pretrained model
    model = ProHMRScene(cfg=model_cfg, device=device,
                               with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                               scene_feat_dim=512, scene_cano=args.scene_cano)
    weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    weights_copy = {}
    weights_copy['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] != 'smpl'}
    model.load_state_dict(weights_copy['state_dict'], strict=False)
    model.eval()
    print('load traind mode from: ', args.checkpoint)


    ################################ Create dataset and data loader
    test_dataset = DatasetEgobody(cfg=model_cfg, train=False, device=device, data_root=args.dataset_root,
                                            dataset_file=os.path.join(args.dataset_root, 'annotation_egocentric_smpl_npz/egocapture_test_smpl.npz'), spacing=1,
                                            add_scale=args.add_bbox_scale, split='test',
                                            scene_type=args.scene_type,
                                            scene_cano=args.scene_cano)
    dataloader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)


    ################################# load smpl models
    smpl_neutral = smplx.create('data/smpl', model_type='smpl', gender='neutral', create_transl=False, batch_size=args.batch_size).to(device)
    smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male', batch_size=args.batch_size).to(device)
    smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female', batch_size=args.batch_size).to(device)


    ################################# create list for full-body evaluation metrics (for mode prediction, z=0)
    g_mpjpe = np.zeros(len(test_dataset))
    mpjpe = np.zeros(len(test_dataset))
    pa_mpjpe = np.zeros(len(test_dataset))
    g_v2v = np.zeros(len(test_dataset))
    v2v = np.zeros(len(test_dataset))  # translation/pelv-aligned
    pa_v2v = np.zeros(len(test_dataset))  # procrustes aligned

    pred_cam_full_list = []


    ################################# create dir to save rendered results
    # cannot render and visualize in 3d at the same time
    if args.render and args.vis_o3d:
        print('[ERROR]: cannot set both args.render and args.vis_o3d to True!')
        exit()
    if args.render:
        model_id = args.checkpoint.split('/')[-2]
        output_vis_folder = os.path.join(args.output_render_root, 'output_prohmr_scene_{}'.format(model_id))
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

    ################################# test of proHMR-scene baseline
    for step, batch in enumerate(tqdm(dataloader)):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        ###### get gt annotations
        gt_pose = {}
        gt_pose['global_orient'] = batch['smpl_params']['global_orient'].to(device)
        gt_pose['transl'] = batch['smpl_params']['transl'].to(device)
        gt_pose['body_pose'] = batch['smpl_params']['body_pose'].to(device)
        gt_pose['betas'] = batch['smpl_params']['betas'].to(device)
        gender = batch['gender'].to(device)
        curr_batch_size = batch['img'].shape[0]
        # img_names = batch['imgname']

        ###### prepare camera params for rendering in full image
        bbox_size = batch['box_size'].to(device)  # [bs]
        bbox_center = batch['box_center'].to(device)  # [bs, 2]
        focal_length = batch['fx'] * model_cfg.CAM.FX_NORM_COEFF  # [bs]
        cam_cx = batch['cam_cx']
        cam_cy = batch['cam_cy']

        if curr_batch_size != args.batch_size:
            smpl_neutral = smplx.create('data/smpl', model_type='smpl', gender='neutral', create_transl=False, batch_size=curr_batch_size).to(device)
            smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male', batch_size=curr_batch_size).to(device)
            smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female', batch_size=curr_batch_size).to(device)

        ###### get pred smpl params
        pred_betas = out['pred_smpl_params']['betas']  # [bs, num_sample, 10]
        pred_body_pose = out['pred_smpl_params']['body_pose']  # [bs, num_sample, 23, 3, 3]
        pred_global_orient = out['pred_smpl_params']['global_orient']  # [bs, num_sample, 1, 3, 3]
        pred_cam = out['pred_cam']  # [bs, num_sample, 3]

        ###### get pred smpl joints / vertices
        pred_output = smpl_neutral(betas=pred_betas.reshape(-1, 10), body_pose=pred_body_pose.reshape(-1, 23, 3, 3),
                                   global_orient=pred_global_orient.reshape(-1, 1, 3, 3), pose2rot=False)
        pred_vertices = pred_output.vertices.reshape(curr_batch_size, -1, 6890, 3)
        pred_keypoints_3d = pred_output.joints.reshape(curr_batch_size, -1, 45, 3)[:, :, :24, :]  # [bs, n_sample, 24, 3]
        pred_vertices_mode = pred_vertices[:, 0]
        pred_keypoints_3d_mode = pred_keypoints_3d[:, 0]  # [bs, 24, 3]

        ###### for mode prediction with z=0
        pred_pelvis_mode = pred_keypoints_3d_mode[:, [0], :].clone()
        pred_keypoints_3d_mode_align = pred_keypoints_3d_mode - pred_pelvis_mode
        pred_vertices_mode_align = pred_vertices_mode - pred_pelvis_mode
        pred_cam_mode = pred_cam[:, 0]

        ##### get pred cam / joints / vertices in full img coord (for mode z=0)
        pred_cam_full = convert_pare_to_full_img_cam(pare_cam=pred_cam_mode, bbox_height=bbox_size,
                                                     bbox_center=bbox_center,
                                                     img_w=cam_cx * 2, img_h=cam_cy * 2,
                                                     focal_length=focal_length,
                                                     crop_res=model_cfg.MODEL.IMAGE_SIZE)  # [bs, 3]
        pred_vertices_mode_full = pred_vertices_mode + pred_cam_full.unsqueeze(1)  # [bs, 6890, 3]
        pred_vertices_full = pred_vertices + pred_cam_full.unsqueeze(1).unsqueeze(1)  # [bs, n_sample, 6890, 3]
        pred_keypoints_3d_mode_full = pred_keypoints_3d_mode + pred_cam_full.unsqueeze(1)  # [bs, 24, 3]
        pred_cam_full_list.append(pred_cam_full)

        ##### get gt body joints / vertices
        gt_body = smpl_male(**gt_pose)
        gt_joints = gt_body.joints
        gt_vertices = gt_body.vertices
        gt_body_female = smpl_female(**gt_pose)
        gt_joints_female = gt_body_female.joints
        gt_vertices_female = gt_body_female.vertices
        gt_joints[gender == 1, :, :] = gt_joints_female[gender == 1, :, :]
        gt_vertices[gender == 1, :, :] = gt_vertices_female[gender == 1, :, :]

        gt_keypoints_3d = gt_joints[:, :24, :]  # [bs, 24, 3] first 24 smpl main body joints
        gt_pelvis = gt_keypoints_3d[:, [0], :].clone()  # [bs,1,3]
        gt_keypoints_3d_align = gt_keypoints_3d - gt_pelvis
        gt_vertices_align = gt_vertices - gt_pelvis

        ######################## visualize 3d bodies and scene in the original physical camera
        if args.vis_o3d and step % args.vis_step == 0:
            # for bi in range(curr_batch_size):  # todo: uncomment if want to visualize for all images within this batch
            for bi in range(1):
                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                pred_body_o3d_full = o3d.geometry.TriangleMesh()
                pred_body_o3d_full.vertices = o3d.utility.Vector3dVector(pred_vertices_mode_full[bi].detach().cpu().numpy())
                pred_body_o3d_full.triangles = o3d.utility.Vector3iVector(smpl_neutral.faces)
                pred_body_o3d_full.compute_vertex_normals()

                if args.vis_o3d_gt:
                    gt_body_o3d_full = o3d.geometry.TriangleMesh()
                    gt_body_o3d_full.vertices = o3d.utility.Vector3dVector(gt_vertices[bi].detach().cpu().numpy())  # [6890, 3]
                    gt_body_o3d_full.triangles = o3d.utility.Vector3iVector(smpl_neutral.faces)
                    gt_body_o3d_full.paint_uniform_color([0, 0, 1])
                    gt_body_o3d_full.compute_vertex_normals()

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
                add_trans = np.array([[1.0, 0, 0, 0],
                                      [0, -1, 0, 0],
                                      [0, 0, -1, 0],
                                      [0, 0, 0, 1]])

                scene_mesh_full.transform(trans_scene_to_main)
                scene_mesh_full.transform(trans_kinect2holo)
                scene_mesh_full.transform(cur_world2pv_transform)
                scene_mesh_full.transform(add_trans)

                ################ visualize
                if args.vis_o3d_gt:
                    o3d.visualization.draw_geometries([scene_pcd, scene_mesh_full, mesh_frame, pred_body_o3d_full, gt_body_o3d_full])
                else:
                    o3d.visualization.draw_geometries([scene_pcd, scene_mesh_full, mesh_frame, pred_body_o3d_full])


        ################################# calculate full-body evaluation metrics (for mode prediction, z=0)
        # G-MPJPE
        error_per_joint = torch.sqrt(((pred_keypoints_3d_mode_full - gt_keypoints_3d) ** 2).sum(dim=-1))
        error = error_per_joint.mean(dim=-1).cpu().numpy()
        g_mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # MPJPE
        error_per_joint = torch.sqrt(((pred_keypoints_3d_mode_align - gt_keypoints_3d_align) ** 2).sum(dim=-1))
        error = error_per_joint.mean(dim=-1).cpu().numpy()
        mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # PA-MPJPE
        pa_error_per_joint = reconstruction_error(pred_keypoints_3d_mode_align.cpu().numpy(),
                                                  gt_keypoints_3d_align.cpu().numpy(), avg_joint=False)  # [bs, n_joints]
        pa_error = pa_error_per_joint.mean(axis=-1)  # [bs]
        pa_mpjpe[step * args.batch_size:step * args.batch_size + curr_batch_size] = pa_error

        # G-V2V
        error_per_verts = torch.sqrt(((pred_vertices_mode_full - gt_vertices) ** 2).sum(dim=-1))
        error = error_per_verts.mean(dim=-1).cpu().numpy()
        g_v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # V2V
        error_per_verts = torch.sqrt(((pred_vertices_mode_align - gt_vertices_align) ** 2).sum(dim=-1))
        error = error_per_verts.mean(dim=-1).cpu().numpy()
        v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = error

        # PA-V2V
        pa_error = reconstruction_error(pred_vertices_mode_align.cpu().numpy(), gt_vertices_align.cpu().numpy(),
                                        avg_joint=True)
        pa_v2v[step * args.batch_size:step * args.batch_size + curr_batch_size] = pa_error

        if step % args.log_freq == 0 and step > 0:
            print('G-MPJPE: ' + str(1000 * g_mpjpe[:step * args.batch_size].mean()))
            print('MPJPE: ' + str(1000 * mpjpe[:step * args.batch_size].mean()))
            print('PA-MPJPE: ' + str(1000 * pa_mpjpe[:step * args.batch_size].mean()))
            print('G-V2V: ' + str(1000 * g_v2v[:step * args.batch_size].mean()))
            print('V2V: ' + str(1000 * v2v[:step * args.batch_size].mean()))
            print('PA-V2V: ' + str(1000 * pa_v2v[:step * args.batch_size].mean()))

        ################################# render pred body on input images
        ##### vis_multi_sample=True: render for all samples, otherwise only render the mode prediction
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
                    # full_img_cam = pred_cam_full[[bi]].detach().cpu().numpy()  # [1, 3] todo: need to debug
                    # pred_vertices_select = pred_vertices[[bi], k]  # [1, n_verts, 3]
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


    print('*** Final Results (full body accuracy for mode z=0) ***')
    print('G-MPJPE: ' + str(1000 * g_mpjpe.mean()))
    print('MPJPE: ' + str(1000 * mpjpe.mean()))
    print('PA-MPJPE: ' + str(1000 * pa_mpjpe.mean()))
    print('G-V2V: ' + str(1000 * g_v2v.mean()))
    print('V2V: ' + str(1000 * v2v.mean()))
    print('PA-V2V: ' + str(1000 * pa_v2v.mean()))


    if args.save_pred_transl:
        pred_cam_full_list = torch.cat(pred_cam_full_list, dim=0).detach().cpu().numpy()

        model_id = args.checkpoint.split('/')[-2]
        output_res_folder = '{}/output_prohmr_scene_{}'.format(args.save_root, model_id)
        os.makedirs(output_res_folder) if not os.path.exists(output_res_folder) else None
        output_res_dict = {}
        output_res_dict['pred_cam_full_list'] = pred_cam_full_list
        with open(os.path.join(output_res_folder, 'results.pkl'), 'wb') as result_file:
            pkl.dump(output_res_dict, result_file)
        print('[INFO] pred transl saved to {}.'.format(output_res_folder))



if __name__ == '__main__':
    test()


