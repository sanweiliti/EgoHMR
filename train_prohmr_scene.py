import argparse
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate
import shutil
import random
import sys
from tensorboardX import SummaryWriter

from configs import get_config
from models.prohmr.prohmr_scene import ProHMRScene
from dataloaders.egobody_dataset import DatasetEgobody
from dataloaders.mocap_dataset import MoCapDataset
from utils.other_utils import *


parser = argparse.ArgumentParser(description='ProHMR-scene training code')
parser.add_argument('--gpu_id', type=int, default='0')
parser.add_argument('--load_pretrained', default='True', type=lambda x: x.lower() in ['true', '1'], help='load pretrained prohmr checkpoint')
parser.add_argument('--load_only_backbone', default='True', type=lambda x: x.lower() in ['true', '1'], help='only load the image encoder in pretrained model')
parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoints_prohmr/checkpoint.pt', help='path to pretrained proHMR checkpoint')
parser.add_argument('--model_cfg', type=str, default='configs/prohmr.yaml', help='Path to config file')
parser.add_argument('--save_dir', type=str, default='runs_try', help='path to save train logs and models')
parser.add_argument('--dataset_root', type=str, default='/mnt/ssd/egobody_release', help='path to egobody dataset')

parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=4, help='# of dataloader num_workers')
parser.add_argument('--num_epoch', type=int, default=100000, help='# of training epochs')
parser.add_argument("--log_step", default=1000, type=int, help='log train losses after n iters')
parser.add_argument("--val_step", default=2000, type=int, help='run validation after n iters')
parser.add_argument("--save_step", default=2000, type=int, help='save models after n iters')

parser.add_argument('--scene_cano', default='False', type=lambda x: x.lower() in ['true', '1'], help='translate scene points to be human-centric')
parser.add_argument('--scene_type', type=str, default='whole_scene', choices=['whole_scene', 'cube'],
                    help='whole_scene (all scene vertices in front of camera) / cube (a 2x2 scene cube around the body)')

parser.add_argument('--with_focal_length', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true focal length as input')
parser.add_argument('--with_cam_center', default='True', type=lambda x: x.lower() in ['true', '1'], help='take true camera center as input')
parser.add_argument('--with_bbox_info', default='True', type=lambda x: x.lower() in ['true', '1'], help='take bbox info as input')

parser.add_argument('--with_full_2d_loss', default='True', type=lambda x: x.lower() in ['true', '1'], help='train with 2d joint loss in full image')
parser.add_argument('--with_global_3d_loss', default='True', type=lambda x: x.lower() in ['true', '1'], help='train with 3d joints loss in global coord')

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


def train(writer, logger):
    model_cfg = get_config(args.model_cfg)

    train_dataset = DatasetEgobody(cfg=model_cfg, train=True, device=device, data_root=args.dataset_root,
                                   dataset_file=os.path.join(args.dataset_root, 'annotation_egocentric_smpl_npz/egocapture_train_smpl.npz'),
                                   add_scale=args.add_bbox_scale,
                                   do_augment=args.do_augment,
                                   split='train',
                                   scene_type=args.scene_type,
                                   scene_cano=args.scene_cano)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers, collate_fn=collate_fn)
    train_dataloader_iter = iter(train_dataloader)

    val_dataset = DatasetEgobody(cfg=model_cfg, train=False, device=device, data_root=args.dataset_root,
                                 dataset_file=os.path.join(args.dataset_root, 'annotation_egocentric_smpl_npz/egocapture_val_smpl.npz'),
                                 spacing=1, add_scale=args.add_bbox_scale, split='val',
                                 scene_type=args.scene_type,
                                 scene_cano=args.scene_cano)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    mocap_dataset = MoCapDataset(dataset_file='data/datasets/cmu_mocap.npz')
    mocap_dataloader = torch.utils.data.DataLoader(mocap_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    mocap_dataloader_iter = iter(mocap_dataloader)

    # Setup model
    model = ProHMRScene(cfg=model_cfg, device=device,
                        with_focal_length=args.with_focal_length, with_bbox_info=args.with_bbox_info, with_cam_center=args.with_cam_center,
                        with_full_2d_loss=args.with_full_2d_loss, with_global_3d_loss=args.with_global_3d_loss,
                        scene_feat_dim=512, scene_cano=args.scene_cano)
    model.train()
    if args.load_pretrained:
        weights = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        if args.load_only_backbone:
            weights_backbone = {}
            weights_backbone['state_dict'] = {k: v for k, v in weights['state_dict'].items() if k.split('.')[0] == 'backbone'}
            model.load_state_dict(weights_backbone['state_dict'], strict=False)
        else:
            model.load_state_dict(weights['state_dict'], strict=False)
        print('[INFO] pretrained model loaded from {}.'.format(args.checkpoint))
        print('[INFO] load_only_backbone: {}'.format(args.load_only_backbone))


    # optimizer
    model.init_optimizers()

    ################################## start training #########################################
    total_steps = 0
    best_loss_keypoints_3d_mode = 10000
    for epoch in range(args.num_epoch):
        for step in tqdm(range(train_dataset.dataset_len // args.batch_size)):
            total_steps += 1
            ### iter over train loader and mocap data loader
            try:
                batch = next(train_dataloader_iter)
            except StopIteration:
                train_dataloader_iter = iter(train_dataloader)
                batch = next(train_dataloader_iter)

            try:
                mocap_batch = next(mocap_dataloader_iter)
            except StopIteration:
                mocap_dataloader_iter = iter(mocap_dataloader)
                mocap_batch = next(mocap_dataloader_iter)

            for param_name in batch.keys():
                if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                    batch[param_name] = batch[param_name].to(device)
            for param_name in batch['smpl_params'].keys():
                batch['smpl_params'][param_name] = batch['smpl_params'][param_name].to(device)

            for param_name in mocap_batch.keys():
                mocap_batch[param_name] = mocap_batch[param_name].to(device)

            ####################### train forward pass ############################
            output = model.training_step(batch, mocap_batch)

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
                with torch.no_grad():
                    for test_step, test_batch in tqdm(enumerate(val_dataloader)):
                        for param_name in test_batch.keys():
                            if param_name not in ['imgname', 'smpl_params', 'has_smpl_params', 'smpl_params_is_axis_angle']:
                                test_batch[param_name] = test_batch[param_name].to(device)
                        for param_name in test_batch['smpl_params'].keys():
                            test_batch['smpl_params'][param_name] = test_batch['smpl_params'][param_name].to(device)

                        ###### validation forward pass
                        val_output = model.validation_step(test_batch)

                        for key in val_output['losses'].keys():
                            if test_step == 0:
                                val_loss_dict[key] = val_output['losses'][key].detach().clone()
                            else:
                                val_loss_dict[key] += val_output['losses'][key].detach().clone()

                for key in val_loss_dict.keys():
                    val_loss_dict[key] = val_loss_dict[key] / test_step
                    writer.add_scalar('val/{}'.format(key), val_loss_dict[key].item(), total_steps)
                    print_str = '[Step {:d}/ Epoch {:d}] [test]  {}: {:.10f}'. \
                        format(step, epoch, key, val_loss_dict[key].item())
                    logger.info(print_str)
                    print(print_str)

                # save model with best loss_keypoints_3d_mode
                if val_loss_dict['loss_keypoints_3d_mode'] < best_loss_keypoints_3d_mode:
                    best_loss_keypoints_3d_mode = val_loss_dict['loss_keypoints_3d_mode']
                    save_path = os.path.join(writer.file_writer.get_logdir(), "best_model.pt")
                    state = {
                        "state_dict": model.state_dict(),
                    }
                    torch.save(state, save_path)
                    logger.info('[*] best model saved\n')
                    print('[*] best model saved\n')

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
    ########## set up writter, logger
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))
    sys.stdout.flush()
    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    shutil.copyfile(args.model_cfg, os.path.join(logdir, args.model_cfg.split('/')[-1]))

    train(writer, logger)





