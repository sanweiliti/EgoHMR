# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch
import torch as th
from copy import deepcopy
# from diffusion.nn import mean_flat, sum_flat
# from diffusion.losses import normal_kl, discretized_gaussian_log_likelihood
# from data_loaders.humanml.scripts import motion_process

from utils.geometry import aa_to_rotmat, rotmat_to_rot6d

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1.):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        rescale_timesteps=False,
        body_rep_mean=None, body_rep_std=None,
    ):
        self.rescale_timesteps = rescale_timesteps  # False
        self.body_rep_mean = body_rep_mean
        self.body_rep_std = body_rep_std

        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )  # [1000]
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )  # [1000]
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, batch, x, t, clip_denoised=True, denoised_fn=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x  ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        B, C = x.shape[:2]  # B: bs  C: 144
        assert t.shape == (B,)
        batch['x_t'] = x
        output_dict = model(batch, self._scale_timesteps(t))
        diffuse_output = output_dict['pred_x_start']  # model_output: # [bs, 144]

        model_variance, model_log_variance = self.posterior_variance, self.posterior_log_variance_clipped
        model_variance = _extract_into_tensor(model_variance, t, x.shape)  # [bs, 144]
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        pred_xstart = diffuse_output
        # for last denoising step (t=0), model_mean=pred_xstart
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "other_outputs": output_dict,
        }


    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


    def p_sample(
        self,
        model,
        batch,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_grad_weight=0.0,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            batch,
            x,
            t,
            clip_denoised=clip_denoised,  # False
            denoised_fn=denoised_fn,  # None
        )  # out: mean, variance, log_variance, pred_xstart, other_outputs(output dict (smpl params, keypoints, etc.) from model)
        noise = th.randn_like(x)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise  # sample x_{t-1} from q(x_{t-1}|x_t, x_0)~N(x_{t-1}, mu(x_t, x_0), ...)
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "other_outputs": out['other_outputs']}


    def p_sample_with_grad(
        self,
        model,
        batch,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_grad_weight=1.0,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            batch,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        if t[0] <= 10:
            grad_coap = model.guide_coll(batch, out['other_outputs'], t, compute_grad='x_t')  # [bs, 144]
            if t[0] >= 5:
                out["mean"] = out["mean"].float() + cond_grad_weight * out['variance'] * grad_coap.float()
            else:
                # hard-coded, fixed weight for last 5 denoising steps
                # otherwise out['variance'] becomes too small at the end of denoising steps
                out["mean"] = out["mean"].float() + cond_grad_weight * 0.01 * grad_coap.float()

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "other_outputs": out['other_outputs']}


    def p_sample_loop(
        self,
        model,
        batch,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_data=None,
        cond_fn_with_grad=False,
        cond_grad_weight=1.0,
        dump_steps=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples.
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []

        for i, sample in enumerate(self.p_sample_loop_progressive(
            model,
            batch,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_data=init_data,
            cond_fn_with_grad=cond_fn_with_grad,
            cond_grad_weight=cond_grad_weight,
        )):
            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(sample["sample"]))
            final = sample  # dict: sample, pred_xstart, other_outputs (output dict from the model for smpl params, keypoints, etc.)
        if dump_steps is not None:
            return dump
        return final


    def p_sample_loop_progressive(
        self,
        model,
        batch,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_data=None,
        cond_fn_with_grad=False,
        cond_grad_weight=1.0,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            data = noise
        else:
            data = th.randn(*shape, device=device)

        if skip_timesteps and init_data is None:  # skip_timesteps: 0, init_data: None
            init_data = th.zeros_like(data)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]  # [999, 998, 997, ..., 1, 0]

        if init_data is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            data = self.q_sample(init_data, my_t, data)

        if progress:  # True
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        # data: x_{t-1} or noise (t=T)
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)  # [bs]
            with th.no_grad():
                sample_fn = self.p_sample_with_grad if cond_fn_with_grad else self.p_sample
                out = sample_fn(
                    model,
                    batch,
                    data,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_grad_weight=cond_grad_weight,
                )  # out: sample, pred_xstart, other_outputs (output dict from the model for smpl params, keypoints, etc.)
                yield out
                data = out["sample"]  # sampled x_{t-1}


    def ddim_sample(
        self,
        model,
        batch,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            batch,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "other_outputs": out['other_outputs']}


    def ddim_sample_with_grad(
        self,
        model,
        batch,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        eta=0.0,
    ):
        out = self.p_mean_variance(
            model,
            batch,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
        )

        ############## different from ddim_sample()
        # the ddim sampling does not work well the the collision-score guidance
        # grad = model.guide_coll(batch, out['other_outputs'], t)  # [bs, 144]
        if t[0] <= 3:
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

            grad = model.guide_coll(batch, out['other_outputs'], t)  # [bs, 144]
            scale = 1.0  # hypeparameter
            eps = eps - (1 - alpha_bar).sqrt() * grad * scale
            out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
            out["mean"], _, _ = self.q_posterior_mean_variance(
                x_start=out["pred_xstart"], x_t=x, t=t
            )
            out["pred_xstart"] = out["pred_xstart"].detach()
        ############## different from ddim_sample()

        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"], "other_outputs": out['other_outputs']}



    def ddim_sample_loop(
        self,
        model,
        batch,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_data=None,
        cond_fn_with_grad=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            batch,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            # cond_fn=cond_fn,
            # model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            skip_timesteps=skip_timesteps,
            init_data=init_data,
            # randomize_class=randomize_class,
            cond_fn_with_grad=cond_fn_with_grad,
        ):
            final = sample
        # return final["sample"]
        return final


    def ddim_sample_loop_progressive(
        self,
        model,
        batch,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        device=None,
        progress=False,
        eta=0.0,
        skip_timesteps=0,
        init_data=None,
        cond_fn_with_grad=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            data = noise
        else:
            data = th.randn(*shape, device=device)

        if skip_timesteps and init_data is None:
            init_data = th.zeros_like(data)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_data is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            data = self.q_sample(init_data, my_t, data)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                sample_fn = self.ddim_sample_with_grad if cond_fn_with_grad else self.ddim_sample
                out = sample_fn(
                    model,
                    batch,
                    data,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    eta=eta,
                )  # out: sample, pred_xstart, other_outputs (output dict from the model for smpl params, keypoints, etc.)
                yield out
                data = out["sample"]


    def training_losses(self, model, batch, t, cur_epoch=0, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the tensor of inputs.
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: predicted/denoised SMPL params / joints, losses, etc.
        """
        batch_size = batch['img'].shape[0]
        full_pose_aa = torch.cat([batch['smpl_params']['global_orient'], batch['smpl_params']['body_pose']],
                                 dim=1).reshape(batch_size, -1, 3)  # [bs, 24, 3]
        full_pose_rotmat = aa_to_rotmat(full_pose_aa.reshape(-1, 3)).view(batch_size, -1, 3, 3)  # [bs, 24, 3, 3]
        full_pose_rot6d = rotmat_to_rot6d(full_pose_rotmat.reshape(-1, 3, 3), rot6d_mode='diffusion').reshape(batch_size, -1, 6)  # [bs, 24, 6]
        x_start = full_pose_rot6d.reshape(batch_size, -1)  # [bs, 144]
        x_start = (x_start - self.body_rep_mean) / self.body_rep_std

        ### add noise to x_start
        if noise is None:
            noise = th.randn_like(x_start)  # same shape as x_start
        x_t = self.q_sample(x_start, t, noise=noise)  # [bs, 263, 1, 196], noised version of clean data x_0 at diffusion timestep t
        batch['x_t'] = x_t
        ### forward a training step and calculate losses
        model_output = model.model.training_step(batch=batch, timesteps=self._scale_timesteps(t), cur_epoch=cur_epoch)  # [bs, 263, 1, 196]
        return model_output


    def val_losses(self,
                   model,
                   batch,
                   shape,
                   clip_denoised=True,
                   progress=False,
                   cond_fn_with_grad=False,
                   cond_grad_weight=1.0,
                   cur_epoch=0,
                   timestep_respacing='',
                   compute_loss=True,):
        # set model to eval() mode
        model.validation_setup()
        # val_output: dict, keys:
        # 'sample', 'pred_xstart', 'other_outputs' (output dict from the model for smpl params, keypoints, etc.)
        if timestep_respacing == '':
            # ddpm
            val_output = self.p_sample_loop(model=model, batch=batch, shape=shape, progress=progress,
                                            clip_denoised=clip_denoised,
                                            cond_fn_with_grad=cond_fn_with_grad, cond_grad_weight=cond_grad_weight)
        elif timestep_respacing[0:4] == 'ddim':
            val_output = self.ddim_sample_loop(model=model, batch=batch, shape=shape, progress=progress,
                                               clip_denoised=clip_denoised, eta=0.0,
                                               cond_fn_with_grad=cond_fn_with_grad)
        else:
            print('timestep_respacing_eval not setup correctly')
            exit()

        if compute_loss:
            val_loss_total = model.compute_loss(batch, val_output['other_outputs'], cur_epoch=cur_epoch)
        # losses in dict val_output['other_outputs']['losses']
        return val_output['other_outputs']



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
