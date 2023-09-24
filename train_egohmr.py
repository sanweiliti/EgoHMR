import argparse
import torch
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import shutil
import random
import sys
from tensorboardX import SummaryWriter

from configs import get_config
from dataloaders.egobody_dataset import DatasetEgobody
from models.egohmr.egohmr import EgoHMR
from diffusion.model_util import create_gaussian_diffusion
from diffusion.resample import UniformSampler
from utils.other_utils import *

parser = argparse.ArgumentParser(description='EgoHMR training code')
parser.add_argument('--gpu_id', type=int, default='0')
parser.add_argument('--load_pretrained_img_enc', default='True', type=lambda x: x.lower() in ['true', '1'], help='load pretraind image encoder from prohmr checkpoint')
parser.add_argument('--checkpoint_img_enc', type=str, default='checkpoints/checkpoints_prohmr/checkpoint.pt', help='prohmr pretrained checkpoint')

parser.add_argument('--model_cfg', type=str, default='configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--save_dir', type=str, default='runs_try_try', help='path to save train logs and models')
parser.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4, help='# of dataloadeer num_workers')
parser.add_argument('--num_epoch', type=int, default=100000, help='# of training epochs ')
parser.add_argument("--log_step", default=1000, type=int, help='log after n iters')
parser.add_argument("--val_step", default=1000, type=int, help='log after n iters')
parser.add_argument("--save_step", default=2000, type=int, help='save models after n iters')

###### scene args
parser.add_argument('--scene_cano', default='True', type=lambda x: x.lower() in ['true', '1'], help='transl scene points to be human-centric')
parser.add_argument('--scene_type', type=str, default='cube', choices=['whole_scene', 'cube'],
                    help='whole_scene (all scene vertices in front of camera) / cube (a 2x2 cube around the body)')

###### traning loss weights
parser.add_argument('--weight_loss_v2v', type=float, default=0.5)
parser.add_argument('--weight_loss_keypoints_3d', type=float, default=0.05)
parser.add_argument('--weight_loss_keypoints_3d_full', type=float, default=0.02)
parser.add_argument('--weight_loss_keypoints_2d_full', type=float, default=0.01)
parser.add_argument('--weight_loss_betas', type=float, default=0.0005)
parser.add_argument('--weight_loss_body_pose', type=float, default=0.001)
parser.add_argument('--weight_loss_global_orient', type=float, default=0.001)
parser.add_argument('--weight_loss_pose_6d_ortho', type=float, default=0.1)
parser.add_argument('--weight_coap_penetration', type=float, default=0.0002)
parser.add_argument('--start_coap_epoch', type=int, default=3, help='from which epoch to use coap scene collision loss')

#### diffusion model args
parser.add_argument("--num_diffusion_timesteps", default=50, type=int, help='total steps for diffusion')
parser.add_argument('--timestep_respacing_eval', type=str, default='ddim5', choices=['ddim5', 'ddpm'], help='ddim/ddpm sampling schedule')
parser.add_argument("--eval_spacing", default=20, type=int, help='downsample val set by # for faster evaluation during training')
parser.add_argument('--cond_mask_prob', type=float, default=0.01, help='by what prob to mask out conditions during training')
parser.add_argument('--only_mask_img_cond', default='True', type=lambda x: x.lower() in ['true', '1'],
                    help='only mask img features during trainig with cond_mask_prob')
parser.add_argument('--pelvis_vis_loosen', default='False', type=lambda x: x.lower() in ['true', '1'],
                    help='set pelvis joint visibility the same as knees, allows more flexibility for lower body diversity')

parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true focal length as input')
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true camera center as input')
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'], help='take bbox info as input')
parser.add_argument('--add_bbox_scale', type=float, default=1.2, help='scale orig bbox size')
parser.add_argument('--do_augment', default='True', type=lambda x: x.lower() in ['true', '1'], help='perform data augmentation')
parser.add_argument('--shuffle', default='True', type=lambda x: x.lower() in ['true', '1'], help='shuffle in train dataloader')

args = parser.parse_args()


torch.cuda.set_device(args.gpu_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('gpu id:', torch.cuda.current_device())



def collate_fn(item):
    try:
        item = default_collate(item)
    except Exception as e:
        import pdb;
        pdb.set_trace()
    return item



def train(writer, logger, logdir):
    model_cfg = get_config(args.model_cfg)

    # Create dataset and data loader
    train_dataset = DatasetEgobody(cfg=model_cfg, train=True, device=device, data_root=args.dataset_root,
                                   dataset_file=os.path.join(args.dataset_root, 'annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'),
                                   add_scale=args.add_bbox_scale, do_augment=args.do_augment, split='train',
                                   scene_type=args.scene_type, scene_cano=args.scene_cano,
                                   get_diffuse_feature=True, body_rep_stats_dir=logdir)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn)
    train_dataloader_iter = iter(train_dataloader)


    val_dataset = DatasetEgobody(cfg=model_cfg, train=False, device=device, data_root=args.dataset_root,
                                 dataset_file=os.path.join(args.dataset_root, 'annotation_egocentric_smpl_npz/egocapture_val_smpl.npz'),
                                 spacing=args.eval_spacing, add_scale=args.add_bbox_scale, split='val',
                                 scene_type=args.scene_type, scene_cano=args.scene_cano,)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Setup model
    preprocess_stats = np.load(os.path.join(logdir, 'preprocess_stats/preprocess_stats.npz'))
    body_rep_mean = torch.from_numpy(preprocess_stats['Xmean']).float().to(device)
    body_rep_std = torch.from_numpy(preprocess_stats['Xstd']).float().to(device)

    model = EgoHMR(cfg=model_cfg, device=device, body_rep_mean=body_rep_mean, body_rep_std=body_rep_std,
                   with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                   scene_feat_dim=512, scene_type=args.scene_type, scene_cano=args.scene_cano,
                   weight_loss_v2v=args.weight_loss_v2v, weight_loss_keypoints_3d=args.weight_loss_keypoints_3d,
                   weight_loss_keypoints_3d_full=args.weight_loss_keypoints_3d_full, weight_loss_keypoints_2d_full=args.weight_loss_keypoints_2d_full,
                   weight_loss_betas=args.weight_loss_betas, weight_loss_body_pose=args.weight_loss_body_pose,
                   weight_loss_global_orient=args.weight_loss_global_orient, weight_loss_pose_6d_ortho=args.weight_loss_pose_6d_ortho,
                   cond_mask_prob=args.cond_mask_prob, only_mask_img_cond=args.only_mask_img_cond,
                   weight_coap_penetration=args.weight_coap_penetration, start_coap_epoch=args.start_coap_epoch,
                   pelvis_vis_loosen=args.pelvis_vis_loosen)
    diffusion_train = create_gaussian_diffusion(num_diffusion_timesteps=args.num_diffusion_timesteps, timestep_respacing='',
                                                body_rep_mean=body_rep_mean, body_rep_std=body_rep_std)
    schedule_sampler = UniformSampler(diffusion_train)

    if args.timestep_respacing_eval == 'ddpm':
        args.timestep_respacing_eval = ''
    diffusion_eval = create_gaussian_diffusion(num_diffusion_timesteps=args.num_diffusion_timesteps, timestep_respacing=args.timestep_respacing_eval,
                                               body_rep_mean=body_rep_mean, body_rep_std=body_rep_std)


    if args.load_pretrained_img_enc:
        weights = torch.load(args.checkpoint_img_enc, map_location=lambda storage, loc: storage)
        weights_backbone = {}
        weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if
                                          k.split('.')[0] == 'backbone'}
        model.load_state_dict(weights_backbone['state_dict'], strict=False)
        print('[INFO] pretrained img encoder loaded from {}.'.format(args.checkpoint_img_enc))


    # optimizer
    model.init_optimizers()

    ################################## start training #########################################
    total_steps = 0
    best_loss_keypoints_3d, best_loss_keypoints_3d_vis = 10000, 10000
    for epoch in range(args.num_epoch):
        for step in tqdm(range(train_dataset.dataset_len // args.batch_size)):
            total_steps += 1
            ### iter over train loader and mocap data loader
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)

            for param_name in batch.keys():
                if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                    batch[param_name] = batch[param_name].to(device)
            for param_name in batch['smpl_params'].keys():
                batch['smpl_params'][param_name] = batch['smpl_params'][param_name].to(device)

            batch_size = batch['img'].shape[0]
            t, weights = schedule_sampler.sample(batch_size, device)
            output = diffusion_train.training_losses(model, batch, t, epoch)

            ####################### log train loss ############################
            if total_steps % args.log_step == 0:
                for key in output['losses'].keys():
                    writer.add_scalar('train/{}'.format(key), output['losses'][key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [train]  {}: {:.10f}'. \
                        format(step, epoch, key, output['losses'][key].item())
                    logger.info(print_str)
                    print(print_str)

            ####################### log val loss #################################
            if total_steps % args.val_step == 0:
                val_loss_dict = {}
                joint_vis_num = 0
                total_sample_num = 0
                with torch.no_grad():
                    for test_step, test_batch in tqdm(enumerate(val_dataloader)):
                        for param_name in test_batch.keys():
                            if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                                test_batch[param_name] = test_batch[param_name].to(device)
                        for param_name in test_batch['smpl_params'].keys():
                            test_batch['smpl_params'][param_name] = test_batch['smpl_params'][param_name].to(device)

                        # val_output = model.validation_step(test_batch, epoch)
                        shape = list(batch['x_t'].shape)
                        shape[0] = test_batch['img'].shape[0]
                        val_output = diffusion_eval.val_losses(model=model, batch=test_batch, shape=shape,
                                                               progress=False, clip_denoised=False, cur_epoch=epoch,
                                                               timestep_respacing=args.timestep_respacing_eval)

                        joint_vis_num += val_output['joint_vis_num_batch'].item()
                        total_sample_num += test_batch['img'].shape[0]
                        for key in val_output['losses'].keys():
                            if test_step == 0:
                                val_loss_dict[key] = val_output['losses'][key].detach().clone()
                            else:
                                val_loss_dict[key] += val_output['losses'][key].detach().clone()

                val_loss_dict['loss_keypoints_3d_vis'] = 0
                for key in val_loss_dict.keys():
                    if key != 'loss_keypoints_3d_vis_batch_sum':
                        val_loss_dict[key] = val_loss_dict[key] / (test_step + 1)
                    else:
                        val_loss_dict['loss_keypoints_3d_vis'] = val_loss_dict['loss_keypoints_3d_vis_batch_sum'] / joint_vis_num * 1000
                    if key != 'loss_keypoints_3d_vis_batch_sum':
                        writer.add_scalar('val/{}'.format(key), val_loss_dict[key].item(), total_steps)
                        print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'. \
                            format(step, epoch, key, val_loss_dict[key].item())
                        logger.info(print_str)
                        print(print_str)

                if val_loss_dict['loss_keypoints_3d_vis'] < best_loss_keypoints_3d_vis:
                    best_loss_keypoints_3d_vis = val_loss_dict['loss_keypoints_3d_vis']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_model_vis.pt")
                    state = {
                        "state_dict": model.state_dict(),
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best model-vis saved\n')
                    print('[*] best model-vis saved\n')


            ################### save trained model #######################
            if total_steps % args.save_step == 0:
                save_path = os.path.join(writer.file_writer.get_logdir(), "last_model.pt")
                state = {
                    "state_dict": model.state_dict(),
                }
                torch.save(state, save_path)
                logger.info('[*] last model saved\n')
                print('[*] last model saved\n')






if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()
    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    shutil.copyfile(args.model_cfg, os.path.join(logdir, args.model_cfg.split('/')[-1]))

    train(writer, logger, logdir)





