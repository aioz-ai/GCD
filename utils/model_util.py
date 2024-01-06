

from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps


def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args, data, model_type='gdance_style'):
    if model_type == 'gdance_style':
        from model.gdance_style import GDanceStyle
        # from model.edge_gdance_style_local_only import GDanceStyle
        model = GDanceStyle(**get_model_args_gdance_style(args, data))
    else:
        raise ValueError('currently support two model types: [edge, gdance_style]')
    diffusion = create_gaussian_diffusion(model, args)
    return model, diffusion



def get_model_args_gdance_style(args, data):

    cond_mode = 'music'

    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1

    if hasattr(data.dataset, "normalizer"):
        normalizer = data.dataset.normalizer
    else:
        normalizer = None

    # SMPL defaults
    data_rep = 'rot6d'
    nfeats = 151 # 3 (trans) + 24*6 (rots) + 4 (contacts)

    arch = 'trans_dec'

    use_film = args.use_film
    music_extra_token = args.music_extra_token

    use_style = args.use_style #True
    fuse_style_music = args.fuse_style_music

    # Discriminator args
    use_discriminator = args.lambda_contrastive > 0.0




    return {'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'normalizer': normalizer,
            'latent_dim': args.latent_dim, 'ff_size': 1024,  'num_heads': 8,
            'num_layers': args.layers, 'num_music_enc_layers': 2,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob,  'arch': arch,
            'use_film': use_film, 'music_extra_token': music_extra_token,
            'use_style': use_style, 'fuse_style_music': fuse_style_music,
            'dataset': args.dataset,"max_persons": args.max_persons, 'max_seq_len': args.target_seq_len,
            'use_discriminator': use_discriminator, 'disc_num_layers': args.disc_num_layers, 'disc_time_aware': args.disc_time_aware
            }
def create_gaussian_diffusion(model, args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling


    #timestep_respacing = 'ddim50'  # can be used for ddim sampling, .
    timestep_respacing = ''  # can be used for ddim sampling,  don't use in training.
    if hasattr(args, 'timestep_respacing'):
        timestep_respacing = args.timestep_respacing

    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_main = args.lambda_main,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_contrastive=args.lambda_contrastive,
        lambda_diversity=0.,
        rot2xyz = model.rot2xyz
    )