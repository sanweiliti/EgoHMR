import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
import smplx
from coap import attach_coap
from smplx.utils import SMPLOutput

from models.resnet import resnet
from models.respointnet import ResnetPointnet
from models.egohmr.modulated_gcn.modulated_gcn import ModulatedGCN
from models.egohmr.losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from utils.geometry import aa_to_rotmat, perspective_projection, rot6d_to_rotmat
from utils.konia_transform import rotation_matrix_to_angle_axis
from utils.other_utils import SMPL_EDGES


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


class EgoHMR(nn.Module):
    def __init__(self, cfg, device=None,
                 body_rep_mean=None, body_rep_std=None,
                 with_focal_length=False, with_bbox_info=False, with_cam_center=False,
                 scene_feat_dim=512, scene_type='whole_scene', scene_cano=False,
                 weight_loss_v2v=0, weight_loss_keypoints_3d=0, weight_loss_keypoints_3d_full=0, weight_loss_keypoints_2d_full=0,
                 weight_loss_betas=0, weight_loss_body_pose=0, weight_loss_global_orient=0, weight_loss_pose_6d_ortho=0,
                 weight_coap_penetration=0, start_coap_epoch=0,
                 cond_mask_prob=0, only_mask_img_cond=False,
                 diffusion_blk=4, gcn_dropout=0.0, gcn_nonlocal_layer=False, gcn_hid_dim=1024,
                 pelvis_vis_loosen=False,
                 diffuse_fuse=False,
                 ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.with_focal_length = with_focal_length
        self.with_bbox_info = with_bbox_info
        self.with_cam_center = with_cam_center

        self.body_rep_mean = body_rep_mean
        self.body_rep_std = body_rep_std
        self.diffuse_feat_dim = 6  # 6d rotation

        self.cond_mask_prob = cond_mask_prob
        self.only_mask_img_cond = only_mask_img_cond
        self.diffuse_fuse = diffuse_fuse

        self.input_process_out_dim = 512
        self.input_process = InputProcess(self.diffuse_feat_dim, self.input_process_out_dim).to(self.device)
        self.timestep_embed_dim = 512
        self.sequence_pos_encoder = PositionalEncoding(self.timestep_embed_dim, dropout=0.1).to(self.device)
        self.embed_timestep = TimestepEmbedder(self.timestep_embed_dim, self.sequence_pos_encoder).to(self.device)

        ##### img encoder
        self.backbone = resnet(cfg).to(self.device)

        ##### scene encoder
        self.scene_type = scene_type
        self.scene_cano = scene_cano
        self.scene_enc = ResnetPointnet(out_dim=scene_feat_dim, hidden_dim=256).to(self.device)

        ##### body transl encoder
        transl_embed_dim = 128
        self.transl_enc = TranslEnc(in_dim=3, out_dim=transl_embed_dim).to(self.device)

        ##### conditioning feature dim
        context_feats_dim = cfg.MODEL.BACKBONE.OUT_CHANNELS  # img_feature_d=2048
        if self.with_focal_length or self.with_vfov:
            context_feats_dim = context_feats_dim + 1
        if self.with_bbox_info:
            context_feats_dim = context_feats_dim + 3
        if self.with_cam_center:
            context_feats_dim = context_feats_dim + 2
        context_feats_dim = context_feats_dim + scene_feat_dim + transl_embed_dim

        ##### denoiser model
        edges = np.array(SMPL_EDGES, dtype=np.int32)
        data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
        adj_mx = sp.coo_matrix((data, (i, j)), shape=(24, 24), dtype=np.float32)
        # build symmetric adjacency matrix
        adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
        adj_mx = normalize(adj_mx)  # + sp.eye(adj_mx.shape[0]))
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)  # [24, 24]
        adj_mx = adj_mx * (1 - torch.eye(adj_mx.shape[0])) + torch.eye(adj_mx.shape[0])
        adj_mx = adj_mx.to(self.device)
        self.diffusion_model = ModulatedGCN(adj=adj_mx,
                                            in_dim=context_feats_dim + self.input_process_out_dim + self.timestep_embed_dim,
                                            hid_dim=gcn_hid_dim, out_dim=self.diffuse_feat_dim,
                                            num_layers=diffusion_blk, p_dropout=gcn_dropout,
                                            nonlocal_layer=gcn_nonlocal_layer).to(self.device)

        ##### layers to predict beta
        self.beta_layer = FCHeadBeta(in_dim=context_feats_dim, condition_on_pose=False, pose_dim=self.diffuse_feat_dim*24).to(self.device)

        ######### Instantiate SMPL model
        self.smpl = smplx.create('data/smpl', model_type='smpl', gender='neutral').to(self.device)
        self.smpl_male = smplx.create('data/smpl', model_type='smpl', gender='male').to(self.device)
        self.smpl_female = smplx.create('data/smpl', model_type='smpl', gender='female').to(self.device)
        self.smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                                 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]  # smpl joint 24 - openpose joint 0
        if not pelvis_vis_loosen:
            self.openpose_to_smpl = [8, 12, 9, 8, 13, 10, 8, 14, 11, 8, 14, 11, 0 , 5, 2, 0, 5, 2, 6, 3, 7, 4, 7, 4]  # openpose joint 8 - smpl joint 0
        else:
            # pelvis_vis_loosen=True: set smpl joint 1/2 the same visibility as knee joints, loosen the visibility constraint a bit
            self.openpose_to_smpl = [8, 13, 10, 8, 13, 10, 8, 14, 11, 8, 14, 11, 1, 5, 2, 0, 5, 2, 6, 3, 7, 4, 7, 4]

        #### coap setup for human-scene collision detection
        self.smpl = attach_coap(self.smpl, pretrained=True, device=device)
        self.smpl.coap.eval()
        for param in self.smpl.coap.parameters():
            param.requires_grad = False
        self.weight_coap_penetration = weight_coap_penetration
        self.start_coap_epoch = start_coap_epoch

        ##### Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.v2v_loss = nn.L1Loss(reduction='none')
        self.smpl_parameter_loss = ParameterLoss()

        self.weight_loss_v2v = weight_loss_v2v
        self.weight_loss_keypoints_3d = weight_loss_keypoints_3d
        self.weight_loss_keypoints_3d_full = weight_loss_keypoints_3d_full
        self.weight_loss_keypoints_2d_full = weight_loss_keypoints_2d_full
        self.weight_loss_betas = weight_loss_betas
        self.weight_loss_body_pose = weight_loss_body_pose
        self.weight_loss_global_orient = weight_loss_global_orient
        self.weight_loss_pose_6d_ortho = weight_loss_pose_6d_ortho


    def init_optimizers(self):
        self.opt_params = list(self.backbone.parameters()) + list(self.scene_enc.parameters()) + \
                          list(self.transl_enc.parameters()) + list(self.beta_layer.parameters()) + \
                          list(self.diffusion_model.parameters()) + \
                          list(self.embed_timestep.parameters()) + list(self.input_process.parameters())
        self.optimizer = torch.optim.AdamW(params=self.opt_params,
                                           lr=self.cfg.TRAIN.LR,
                                           weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)


    def mask_cond(self, cond, only_mask_img_cond=False, force_mask=False):
        bs, J, d = cond.shape
        if force_mask:
            if only_mask_img_cond:
                mask = torch.ones([bs, J, d]).to(self.device)
                mask[:, :, 0:2048] = 0
                return cond * mask
            else:
                return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond  [bs, 1]
            if only_mask_img_cond:
                # only set image features to 0, then the model is trained only with scene condition
                mask_final = torch.zeros([bs, J, d]).to(self.device)
                mask_final[torch.where(mask==1)[0], :, 0:2048] = 1
                return cond * (1. - mask_final)
            else:
                return cond * (1. - mask)
        else:
            return cond



    def forward(self, batch, timesteps, eval_with_uncond=True):
        # timesteps: [batch_size] (int)
        batch_size = batch['img'].shape[0]

        ############## timestep encoding
        timestep_emb = self.embed_timestep(timesteps).squeeze(0)  # [bs, d]  # timesteps: [bs]
        timestep_emb = timestep_emb.unsqueeze(1).repeat(1, 24, 1)   # [bs, 24, d]

        ############### img encoding
        img = batch['img']  # [bs, 3, 224, 224]
        img_feats = self.backbone(img)  # [bs, 2048]

        ############### visibility info
        vis_mask_openpose = batch['orig_keypoints_2d'][:, :, -1] > 0  # [bs, 25]
        vis_mask_openpose[:, 8] = True  # pelvis: set global R always as visible
        vis_mask_smpl = vis_mask_openpose[:, self.openpose_to_smpl]  # [bs, 24]
        batch['vis_mask_smpl'] = vis_mask_smpl
        img_feats_with_vis_mask = img_feats.unsqueeze(1).repeat(1, 24, 1)  # [bs, 24, 2048]
        img_feats_with_vis_mask = img_feats_with_vis_mask * (vis_mask_smpl.unsqueeze(-1).repeat(1, 1, img_feats.shape[-1]))   # [bs, 24, 2048]


        ############## camera info encoding
        cam_feats = []
        if self.with_focal_length:
            cam_feats = [batch['fx'].unsqueeze(1)] + cam_feats  # [bs, 1]
        if self.with_bbox_info:
            orig_fx = batch['fx'] * self.cfg.CAM.FX_NORM_COEFF
            bbox_info = torch.stack([batch['box_center'][:, 0] / orig_fx, batch['box_center'][:, 1] / orig_fx, batch['box_size'] / orig_fx], dim=-1)  # [bs, 3]
            cam_feats = [bbox_info] + cam_feats   # [bs, 3(+1)]
        if self.with_cam_center:
            orig_fx = batch['fx'] * self.cfg.CAM.FX_NORM_COEFF
            cam_center = torch.stack([batch['cam_cx'] / orig_fx, batch['cam_cy'] / orig_fx], dim=-1)  # [bs, 2]
            cam_feats = [cam_center] + cam_feats  # [bs, 2(+3)(+1)]

        ############## scene encoding
        input_transl = batch['smpl_params']['transl']  # [bs, 3]
        self.input_transl = input_transl
        if self.scene_cano:
            self.scene_pcd_verts = batch['scene_pcd_verts_full'] - input_transl.unsqueeze(1)  # [bs, n_pts, 3]
        else:
            self.scene_pcd_verts = batch['scene_pcd_verts_full']
        scene_feats = self.scene_enc(self.scene_pcd_verts)  # [bs, scene_feat_dim]

        ############## global transl encoding
        transl_feat = self.transl_enc(input_transl)  # [bs, 128]

        ############# get conditioning feature: img(with vis mask) + scene + cam_info + body transl
        conditioning_feats = [scene_feats] + [transl_feat] + cam_feats
        conditioning_feats = torch.cat(conditioning_feats, dim=1)  # [bs, 512+3+1+3+2]
        conditioning_feats = conditioning_feats.unsqueeze(1).repeat(1, 24, 1)  # [bs, 24, 512+3+1+3+2]
        conditioning_feats = torch.cat([img_feats_with_vis_mask, conditioning_feats], dim=-1)  # [bs, 24, 2048+512+3+1+3+2]

        ############### final condition: mask condition with p=cond_mask_prob
        conditioning_feats_masked = self.mask_cond(conditioning_feats,
                                                   only_mask_img_cond=self.only_mask_img_cond,
                                                   force_mask=False)

        ############# pass to denoising model
        output = {}
        x_t = batch['x_t']  # noisy full_pose_rot6d  [bs, 144]
        x_t = x_t.reshape(batch_size, 24, -1)  # [bs, 24, 6]
        x_t_feat = self.input_process(x_t)  # [bs, 24, latent_dim=512]

        diffuse_feat = torch.cat([conditioning_feats_masked, x_t_feat, timestep_emb], axis=-1)  # [bs, 24, (2048+512+3+6)+512+512]
        diffuse_output = self.diffusion_model(diffuse_feat)  # body pose 6d [bs, 24, 6]

        if self.diffuse_fuse:
            if eval_with_uncond:
                # generation with only scene conditions, without image condition
                conditioning_feats_masked_all = self.mask_cond(conditioning_feats,
                                                               only_mask_img_cond=self.only_mask_img_cond,
                                                               force_mask=True)  # mask conditions for all data
                diffuse_feat_uncond = torch.cat([conditioning_feats_masked_all, x_t_feat, timestep_emb], axis=-1)
                diffuse_output_uncond = self.diffusion_model(diffuse_feat_uncond)
                diffuse_output_cond = diffuse_output.clone()
                guidance_param = 0
                diffuse_output = diffuse_output_uncond + guidance_param * (diffuse_output - diffuse_output_uncond)
                # replace joint rotation for visible joints by generated results also conditioned on image feature
                vis_mask_body_pose_6d = vis_mask_smpl.unsqueeze(-1).repeat(1, 1, 6).reshape(batch_size, -1)  # [bs, 144]
                diffuse_output = diffuse_output.reshape(batch_size, -1)
                diffuse_output_cond = diffuse_output_cond.reshape(batch_size, -1)
                diffuse_output[vis_mask_body_pose_6d] = diffuse_output_cond[vis_mask_body_pose_6d]

        diffuse_output = diffuse_output.reshape(batch_size, -1)  # [bs, 144]
        output['pred_x_start'] = diffuse_output
        diffuse_output = diffuse_output * self.body_rep_std + self.body_rep_mean
        pred_pose_6d = diffuse_output
        pred_pose_rotmat = rot6d_to_rotmat(pred_pose_6d, rot6d_mode='diffusion').view(batch_size, 24, 3, 3)  # [bs, 24, 3, 3]

        ############## predict beta
        conditioning_feats_beta = [img_feats] + [scene_feats] + [transl_feat] + cam_feats
        conditioning_feats_beta = torch.cat(conditioning_feats_beta, dim=1)
        pred_betas = self.beta_layer(conditioning_feats_beta, diffuse_output)  # [bs, 10]

        ############## Store useful regression outputs to the output dict
        pred_smpl_params = {'global_orient': pred_pose_rotmat[:, [0]],
                            'body_pose': pred_pose_rotmat[:, 1:],
                            'betas': pred_betas}

        #  global_orient: [bs, 1, 3, 3], body_pose: [bs, 23, 3, 3], shape...
        output['pred_smpl_params'] = {k: v.clone() for k, v in pred_smpl_params.items()}
        output['pred_pose_6d'] = pred_pose_6d

        self.smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, return_full_pose=True, pose2rot=False)
        output['pred_keypoints_3d'] = self.smpl_output.joints  # [bs, 45, 3]
        output['pred_vertices'] = self.smpl_output.vertices  # [bs, 6890, 3]

        ################ get global 3d joints, 2d projections on full image
        dtype = pred_smpl_params['body_pose'].dtype
        if self.with_focal_length:
            focal_length = batch['fx'].unsqueeze(-1) # [bs, 1]
            focal_length = focal_length.repeat(1, 2)  # [bs, 2]
            focal_length = focal_length * self.cfg.CAM.FX_NORM_COEFF
            camera_center_full = torch.cat([batch['cam_cx'].unsqueeze(-1), batch['cam_cy'].unsqueeze(-1)], dim=-1)  # [bs, 2]
        else:
            focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=self.device, dtype=dtype)  # 5000
            camera_center_full = torch.tensor([[[960.0, 540.0]]]).to(self.device).float().repeat(batch_size, 1)  # [bs, 2]
        self.camera_center_full = camera_center_full
        self.focal_length = focal_length

        ##### project predicted 3d joints on full 2d img
        output['pred_keypoints_3d_full'] = output['pred_keypoints_3d'] + input_transl.unsqueeze(1)  # [bs, 45, 3]
        pred_keypoints_2d_full = perspective_projection(output['pred_keypoints_3d'],
                                                        translation=input_transl,
                                                        camera_center=camera_center_full,
                                                        focal_length=focal_length)  # [bs, 45, 2]
        pred_keypoints_2d_full[:, :, 0] = pred_keypoints_2d_full[:, :, 0] / 1920 - 0.5  # in [-0.5, 0.5]
        pred_keypoints_2d_full[:, :, 1] = pred_keypoints_2d_full[:, :, 1] / 1080 - 0.5  # in [-0.5, 0.5]
        output['pred_keypoints_2d_full'] = pred_keypoints_2d_full  # [bs, 45, 2]

        return output



    def compute_loss(self, batch, output, cur_epoch=0):
        pred_smpl_params = output['pred_smpl_params']
        pred_pose_6d = output['pred_pose_6d']
        pred_keypoints_3d = output['pred_keypoints_3d'][:, 0:24]  # [bs, 24, 3]
        pred_keypoints_3d_full = output['pred_keypoints_3d_full'][:, 0:24]

        ### change smpl topology to openpose topology
        pred_keypoints_2d_full = output['pred_keypoints_2d_full']
        pred_keypoints_2d_full = pred_keypoints_2d_full[:, self.smpl_to_openpose, :]

        batch_size = pred_smpl_params['body_pose'].shape[0]

        # Get annotations
        gt_keypoints_2d_full = batch['orig_keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']  # [bs, 24, 3]
        gt_keypoints_3d_full = batch['keypoints_3d_full']
        gt_smpl_params = batch['smpl_params']
        is_axis_angle = batch['smpl_params_is_axis_angle']
        gt_gender = batch['gender']

        ####### Compute 2d/3d keypoint loss
        # 2d joint loss on full img
        loss_keypoints_2d_full = self.keypoint_2d_loss(pred_keypoints_2d_full,
                                                       gt_keypoints_2d_full,
                                                       joints_to_ign=[1, 9, 12]).mean()

        # 3d joint loss in cropped img coord
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d,
                                                  gt_keypoints_3d,
                                                  pelvis_id=0, pelvis_align=True).mean()

        # 3d joint loss in full img coord
        loss_keypoints_3d_full = self.keypoint_3d_loss(pred_keypoints_3d_full,
                                                       gt_keypoints_3d_full,
                                                       pelvis_align=False).mean()

        ####### compute v2v loss
        gt_smpl_output = self.smpl_male(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices = gt_smpl_output.vertices  # smplx vertices
        gt_joints = gt_smpl_output.joints
        gt_smpl_output_female = self.smpl_female(**{k: v.float() for k, v in gt_smpl_params.items()})
        gt_vertices_female = gt_smpl_output_female.vertices
        gt_joints_female = gt_smpl_output_female.joints
        gt_vertices[gt_gender == 1, :, :] = gt_vertices_female[gt_gender == 1, :, :]  # [bs, 6890, 3]
        gt_joints[gt_gender == 1, :, :] = gt_joints_female[gt_gender == 1, :, :]  # [bs, 45, 3]

        pred_vertices = output['pred_vertices']  # [bs, 6890, 3]
        loss_v2v = self.v2v_loss(pred_vertices - pred_keypoints_3d[:, [0], :].clone(),
                                 gt_vertices - gt_joints[:, [0], :].clone()).mean()

        ####### 3d joint loss for visible joints
        loss_keypoints_3d_vis_batch_sum = torch.tensor(0.0).to(self.device)
        if not self.training:
            pred_keypoints_3d_align = pred_keypoints_3d[:, 0:24] - pred_keypoints_3d[:, [0], :].clone()  # [bs, 24, 3]
            gt_keypoints_3d_align = gt_keypoints_3d[:, 0:24] - gt_keypoints_3d[:, [0], :].clone()
            ########## get visbiilty mask of gt body
            gt_keypoints_2d_full_smpl = perspective_projection(gt_joints,
                                                               translation=torch.zeros([batch_size, 3]).to(self.device),
                                                               camera_center=self.camera_center_full,
                                                               focal_length=self.focal_length)  # [bs, 45, 2]
            gt_keypoints_2d_full_smpl = gt_keypoints_2d_full_smpl[:, :24]   # [bs, 24, 2]
            joint_vis_mask = (gt_keypoints_2d_full_smpl[:, :, 0] >= 0) * (gt_keypoints_2d_full_smpl[:, :, 0] < 1920) * \
                             (gt_keypoints_2d_full_smpl[:, :, 1] >= 0) * (gt_keypoints_2d_full_smpl[:, :, 1] < 1080)   # [bs, 24]
            loss_keypoints_3d_vis = (torch.sqrt(((pred_keypoints_3d_align - gt_keypoints_3d_align)**2).sum(dim=-1))) * joint_vis_mask  # [bs, 24]
            loss_keypoints_3d_vis_batch_sum = loss_keypoints_3d_vis.sum()
            joint_vis_num_batch = joint_vis_mask.sum()

        ########### Compute loss on SMPL parameters (pose in rot mat form 3x3)
        # loss_smpl_params: keys: ['global_orient'(bs, n_sample, 1, 3, 3), 'body_pose'(bs, n_sample, 23, 3, 3), 'betas']
        loss_smpl_params = {}
        for k, pred in pred_smpl_params.items():
            if k != 'transl':
                gt = gt_smpl_params[k]
                if is_axis_angle[k].all():
                    gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)  # [bs, 1/23, 3, 3]
                ## MSE loss for rotation/shape
                loss_smpl_params[k] = self.smpl_parameter_loss(pred, gt).sum()/batch_size

        ########### Compute orthonormal loss on 6D representations
        pred_pose_6d = pred_pose_6d.reshape(-1, 3, 2)  # different 6d order from prohmr code
        loss_pose_6d_ortho = (torch.matmul(pred_pose_6d.permute(0, 2, 1), pred_pose_6d) - torch.eye(2, device=self.device, dtype=pred_pose_6d.dtype).unsqueeze(0)) ** 2
        loss_pose_6d_ortho = loss_pose_6d_ortho.mean()

        ########### coap scene collision loss
        loss_coap_penetration = torch.tensor(0.0).to(self.device)
        if self.weight_coap_penetration > 0 and cur_epoch >= self.start_coap_epoch:
            smpl_output_mode = SMPLOutput()
            smpl_output_mode.vertices = self.smpl_output.vertices
            smpl_output_mode.joints = self.smpl_output.joints
            smpl_output_mode.full_pose = self.smpl_output.full_pose  # [bs, 24, 3, 3]
            smpl_output_mode.full_pose = rotation_matrix_to_angle_axis(self.smpl_output.full_pose.reshape(-1, 3, 3)).reshape(batch_size, -1)  # [bs, 24*3]

            smpl_output_mode_list = [SMPLOutput() for _ in range(batch_size)]
            loss_coap_penetration_mode_list = torch.zeros([batch_size]).to(self.device)
            for i in range(batch_size):
                smpl_output_mode_list[i].vertices = smpl_output_mode.vertices[[i]].clone()
                smpl_output_mode_list[i].joints = smpl_output_mode.joints[[i]].clone()
                smpl_output_mode_list[i].full_pose = smpl_output_mode.full_pose[[i]].clone()
                ### sample scene verts
                bb_min = smpl_output_mode_list[i].vertices.min(1).values.reshape(1, 3).detach()
                bb_max = smpl_output_mode_list[i].vertices.max(1).values.reshape(1, 3).detach()
                # print(bb_min, bb_max)
                inds = (self.scene_pcd_verts[[i]] >= bb_min).all(-1) & (self.scene_pcd_verts[[i]] <= bb_max).all(-1)
                #### do not take too many scene points, due to GPU memory limit
                if inds.sum() > 4000:
                    inds[:, 4000:] = False
                # print('sampled point shape:', inds.sum())
                if inds.any():
                    sampled_scene_pcd = self.scene_pcd_verts[[i]][inds].unsqueeze(0)  # [1, sample_verts_num, 3]
                    loss_coap_penetration_mode_list[i] = self.smpl.coap.collision_loss(sampled_scene_pcd,
                                                                                       smpl_output_mode_list[i],
                                                                                       ret_collision_mask=None)
            loss_coap_penetration = torch.mean(loss_coap_penetration_mode_list)


        loss = self.weight_loss_v2v * loss_v2v + \
               self.weight_loss_keypoints_3d * loss_keypoints_3d + \
               self.weight_loss_keypoints_3d_full * loss_keypoints_3d_full + \
               self.weight_loss_keypoints_2d_full * loss_keypoints_2d_full + \
               self.weight_loss_betas * loss_smpl_params['betas'] + \
               self.weight_loss_body_pose * loss_smpl_params['body_pose'] + \
               self.weight_loss_global_orient * loss_smpl_params['global_orient'] + \
               self.weight_loss_pose_6d_ortho * loss_pose_6d_ortho + \
               self.weight_coap_penetration * loss_coap_penetration

        losses = dict(loss=loss.detach(),
                      loss_v2v=loss_v2v.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_keypoints_3d_full=loss_keypoints_3d_full.detach(),
                      loss_keypoints_2d_full=loss_keypoints_2d_full.detach(),
                      loss_betas=loss_smpl_params['betas'].detach(),
                      loss_body_pose=loss_smpl_params['body_pose'].detach(),
                      loss_global_orient=loss_smpl_params['global_orient'].detach(),
                      loss_pose_6d_ortho=loss_pose_6d_ortho.detach(),
                      loss_coap_penetration=loss_coap_penetration.detach(),
                      loss_keypoints_3d_vis_batch_sum=loss_keypoints_3d_vis_batch_sum.detach(),
                      )

        output['losses'] = losses
        if not self.training:
            output['joint_vis_num_batch'] = joint_vis_num_batch

        return loss



    def training_step(self, batch, timesteps, cur_epoch):
        self.training = True
        self.backbone.train()
        self.scene_enc.train()
        self.transl_enc.train()
        self.beta_layer.train()
        self.diffusion_model.train()

        self.input_process.train()
        self.embed_timestep.train()

        ### forward step
        output = self.forward(batch, timesteps, eval_with_uncond=False)
        ### compute loss
        loss = self.compute_loss(batch, output, cur_epoch=cur_epoch)
        ### backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return output


    def validation_setup(self):
        self.training = False
        self.backbone.eval()
        self.scene_enc.eval()
        self.transl_enc.eval()
        self.beta_layer.eval()
        self.diffusion_model.eval()

        self.input_process.eval()
        self.embed_timestep.eval()


    def eval_coll(self, output):
        pred_smpl_params = output['pred_smpl_params']
        batch_size = pred_smpl_params['body_pose'].shape[0]

        smpl_output_mode = SMPLOutput()
        smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, return_full_pose=True, pose2rot=False)
        smpl_output_mode.vertices = smpl_output.vertices
        smpl_output_mode.joints = smpl_output.joints
        smpl_output_mode.full_pose = rotation_matrix_to_angle_axis(smpl_output.full_pose.reshape(-1, 3, 3)).reshape(batch_size, -1)  # [bs, 24*3]
        smpl_output_mode_list = [SMPLOutput() for _ in range(batch_size)]
        coll_ratio_list = []
        for i in range(batch_size):
            smpl_output_mode_list[i].vertices = smpl_output_mode.vertices[[i]].clone()
            smpl_output_mode_list[i].joints = smpl_output_mode.joints[[i]].clone()
            smpl_output_mode_list[i].full_pose = smpl_output_mode.full_pose[[i]].clone()
            ### sample scene verts
            bb_min = smpl_output_mode_list[i].vertices.min(1).values.reshape(1, 3).detach()
            bb_max = smpl_output_mode_list[i].vertices.max(1).values.reshape(1, 3).detach()
            # print(bb_min, bb_max)
            inds = (self.scene_pcd_verts[[i]] >= bb_min).all(-1) & (self.scene_pcd_verts[[i]] <= bb_max).all(-1)
            if inds.any():
                sampled_scene_pcd = self.scene_pcd_verts[[i]][inds].unsqueeze(0)  # [1, sample_verts_num, 3]
                occupancy = self.smpl.coap.query(sampled_scene_pcd, smpl_output_mode_list[i])  # [1, sample_verts_num] >0.5: inside, <0.5, outside
                cur_coll_ratio = (occupancy > 0.5).sum() / self.scene_pcd_verts.shape[1]  # scene verts with collisions
                coll_ratio_list.append(cur_coll_ratio.detach().item())
            else:
                coll_ratio_list.append(0.0)  # no collision
        return coll_ratio_list


    def guide_coll(self, batch, output, t, compute_grad='x_t'):
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = output['pred_x_start']
            x_t = x_t.detach().requires_grad_()

            batch_size = x_t.shape[0]
            smpl_output_mode = SMPLOutput()

            x_t = x_t * self.body_rep_std + self.body_rep_mean
            pred_pose_rotmat = rot6d_to_rotmat(x_t, rot6d_mode='diffusion').view(batch_size, 24, 3, 3)  # [bs, 24, 3, 3]

            ############## Store useful regression outputs
            pred_smpl_params = {'global_orient': pred_pose_rotmat[:, [0]],
                                'body_pose': pred_pose_rotmat[:, 1:],
                                'betas': output['pred_smpl_params']['betas'].detach()}
            # global_orient: [bs, 1, 3, 3], body_pose: [bs, 23, 3, 3], shape...

            smpl_output = self.smpl(**{k: v.float() for k, v in pred_smpl_params.items()}, return_full_pose=True, pose2rot=False)
            smpl_output_mode.vertices = smpl_output.vertices
            smpl_output_mode.joints = smpl_output.joints
            smpl_output_mode.full_pose = rotation_matrix_to_angle_axis(smpl_output.full_pose.reshape(-1, 3, 3)).reshape(batch_size, -1)  # [bs, 24*3]

            smpl_output_mode_list = [SMPLOutput() for _ in range(batch_size)]
            loss_coap_penetration_batch = torch.zeros([batch_size]).to(self.device)
            # due to gpu memory limit, didn't batchwise collision loss
            for i in range(batch_size):
                smpl_output_mode_list[i].vertices = smpl_output_mode.vertices[[i]]
                smpl_output_mode_list[i].joints = smpl_output_mode.joints[[i]]
                smpl_output_mode_list[i].full_pose = smpl_output_mode.full_pose[[i]]
                ### sample scene verts
                bb_min = smpl_output_mode_list[i].vertices.min(1).values.reshape(1, 3).detach()
                bb_max = smpl_output_mode_list[i].vertices.max(1).values.reshape(1, 3).detach()
                inds = (self.scene_pcd_verts[[i]] >= bb_min).all(-1) & (self.scene_pcd_verts[[i]] <= bb_max).all(-1)
                if inds.any():
                    sampled_scene_pcd = self.scene_pcd_verts[[i]][inds]  # [sample_verts_num, 3]
                    loss_coap_penetration_batch[i] = self.smpl.coap.collision_loss(sampled_scene_pcd.unsqueeze(0),
                                                                                    smpl_output_mode_list[i],
                                                                                    ret_collision_mask=None)
                else:
                    loss_coap_penetration_batch[i] = 0.0

            if sum(loss_coap_penetration_batch==0) < len(loss_coap_penetration_batch):
                grad_coap = torch.autograd.grad([-loss_coap_penetration_batch.mean()], [x_t])[0]
                grad_coap = grad_coap.reshape(-1, 24, 6)
                grad_coap[:, 0:3] = grad_coap[:, 0:3] * 1
                grad_coap[:, 3:] = grad_coap[:, 3:] * 2
                # ignore upper body part
                grad_coap[:, [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = 0
                grad_coap = grad_coap.reshape(-1, 144)
            else:
                grad_coap = torch.zeros([batch_size, 144]).to(self.device)
            x_t.detach()

            ######## visualization
            # for i in range(batch_size):
            #     if i > 0:
            #         print('diffusion step =', t[0].item())
            #         print('coap loss cur batch: ', loss_coap_penetration_batch[i])
            #         import open3d as o3d
            #         mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            #         pred_body_o3d = o3d.geometry.TriangleMesh()
            #         pred_body_o3d.vertices = o3d.utility.Vector3dVector(smpl_output_mode_list[i].vertices[0].detach().cpu().numpy())
            #         pred_body_o3d.triangles = o3d.utility.Vector3iVector(self.smpl_male.faces)
            #         pred_body_o3d.compute_vertex_normals()
            #         scene_pcd = o3d.geometry.PointCloud()
            #         scene_pcd.points = o3d.utility.Vector3dVector(self.scene_pcd_verts[i].detach().cpu().numpy())
            #         # scene_normals_pcd = o3d.geometry.PointCloud()
            #         # scene_normals_pcd.points = o3d.utility.Vector3dVector(new_points_sample_2.detach().cpu().numpy())
            #         # color = np.zeros([len(new_points_sample_2), 3])
            #         # color[:, 0] = 1.0
            #         # scene_normals_pcd.colors = o3d.utility.Vector3dVector(color)
            #         import trimesh
            #         trimesh.PointCloud(smpl_output_mode_list[i].vertices.detach().cpu().numpy()[0]).export('noisy_body_{}.ply'.format(i))
            #         o3d.visualization.draw_geometries([scene_pcd, mesh_frame, pred_body_o3d])
            #
            #         # o3d.visualization.draw_geometries([scene_pcd, mesh_frame, pred_body_o3d, scene_normals_pcd])
            #
            #         # transformation = np.identity(4)
            #         # transformation[:3, 3] = smpl_output_mode_list[i].joints[0][4].detach().cpu().numpy()
            #         # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            #         # sphere.paint_uniform_color([70 / 255, 130 / 255, 180 / 255])  # steel blue 70,130,180
            #         # sphere.compute_vertex_normals()
            #         # sphere.transform(transformation)
            #         # o3d.visualization.draw_geometries([scene_pcd, mesh_frame, sphere])

        return grad_coap



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [5000, 1]: 0, 1, 2, ..., 4999
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))  # torch.arange(0, d_model, 2): [256]: 0, 2, 4, 6, 8, ..., 510  div_term: [256]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)   # [5000, 1, 512]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)   # timesteps: [bs]


class InputProcess(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_dim, self.latent_dim)

    def forward(self, x):
        x = self.poseEmbedding(x)
        return x


class FCHeadBeta(nn.Module):
    def __init__(self, in_dim=None, condition_on_pose=False, pose_dim=144):
        super(FCHeadBeta, self).__init__()
        self.condition_on_pose = condition_on_pose
        if self.condition_on_pose:
            in_dim = in_dim + pose_dim
        self.layers = nn.Sequential(nn.Linear(in_dim, 1024),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(1024, 10))
        nn.init.xavier_uniform_(self.layers[2].weight, gain=0.02)

        mean_params = np.load('data/smpl_mean_params.npz')
        init_betas = torch.from_numpy(mean_params['shape'].astype(np.float32))[None]  # [1, 10]
        self.register_buffer('init_betas', init_betas)

    def forward(self, feats, pred_pose):
        if self.condition_on_pose:
            feats = torch.cat([feats, pred_pose], dim=-1)  # [bs, feat_dim+144]
        offset = self.layers(feats)  # [bs, 10]

        pred_betas = offset + self.init_betas
        return pred_betas


class TranslEnc(nn.Module):
    def __init__(self, in_dim=3, out_dim=128):
        super(TranslEnc, self).__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim,64),
                                    nn.ReLU(inplace=False),
                                    nn.Linear(64, out_dim))

    def forward(self, input):
        transl_feat = self.layers(input)
        return transl_feat