import torch
import torch.nn as nn

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d, gt_keypoints_2d, joints_to_ign=None):
        # gt_keypoints_2d: [bs, n_joints, 3]
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()  # [B, N, 1]
        conf[:, joints_to_ign, :] = 0
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2)) # [bs]
        return loss


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d=None, gt_keypoints_3d=None, pelvis_id=0, pelvis_align=False):
        gt_keypoints_3d = gt_keypoints_3d.clone()
        if pelvis_align:
            pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, [pelvis_id], :].clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, [pelvis_id], :]
        loss = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d).sum(dim=(1,2)) # [bs]
        return loss

class Keypoint3DLossMask(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLossMask, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d=None, gt_keypoints_3d=None, pelvis_id=0, pelvis_align=False):
        gt_keypoints_3d = gt_keypoints_3d.clone()
        if pelvis_align:
            pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, [pelvis_id], :].clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_keypoints_3d[:, [pelvis_id], :]
        loss = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d).sum(dim=(1,2)) # [bs]
        return loss


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param, gt_param):
        loss_param = self.loss_fn(pred_param, gt_param)
        return loss_param



class GeodesicLoss(nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def compute_geodesic_distance(self, m1, m2):
        """ Compute the geodesic distance between two rotation matrices.
        Args:
            m1, m2: Two rotation matrices with the shape (batch x 3 x 3).
        Returns:
            The minimal angular difference between two rotation matrices in radian form [0, pi].
        """
        batch = m1.shape[0]
        m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

        cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
        cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
        cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

        theta = torch.acos(cos)

        return theta

    def __call__(self, m1, m2, reduction='mean'):
        loss = self.compute_geodesic_distance(m1, m2)

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError(f'unsupported reduction: {reduction}')



# def cal_bps_body2scene(scene_pcd, body_pcd):
#     # scene_pcd: [bs, n_scene_pts, 3], body_pcd: [bs, 24/45/66/, 3]
#     n_scene_pts = scene_pcd.shape[1]
#     n_body_pts = body_pcd.shape[1]
#     scene_pcd_repeat = scene_pcd.unsqueeze(2).repeat(1, 1, n_body_pts, 1)  # [bs, n_scene_pts, n_body_pts， 3]
#     body_pcd_repeat = body_pcd.unsqueeze(1).repeat(1, n_scene_pts, 1, 1)  # [bs, n_scene_pts, n_body_pts， 3]
#     dist = torch.sum(((scene_pcd_repeat - body_pcd_repeat) ** 2), dim=-1).sqrt()  # [bs, n_scene_pts, n_body_pts]
#     min_dist = torch.min(dist, dim=1).values  # [bs, n_body_pts]
#     return min_dist
