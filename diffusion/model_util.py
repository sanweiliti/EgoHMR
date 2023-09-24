from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(num_diffusion_timesteps=1000, timestep_respacing='ddim5',
                              body_rep_mean=None, body_rep_std=None,
                              ):
    # default params
    steps = num_diffusion_timesteps
    scale_beta = 1.  # no scaling
    rescale_timesteps = False
    noise_schedule = 'cosine'
    betas = gd.get_named_beta_schedule(noise_schedule, steps, scale_beta)  # [time_steps]

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        body_rep_mean=body_rep_mean,
        body_rep_std=body_rep_std,
    )