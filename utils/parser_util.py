from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    ## overwrite model options
    # for a in model_args.keys():
    #     setattr(args, a, model_args[a])

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size during training.")


def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")



def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--model_type", default='gdance_style', choices=['gdance_style'], type=str, help="")
    group.add_argument("--arch", default='trans_dec',
                       choices=['trans_enc', 'trans_dec'], type=str,
                       help="Network Architecture type.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer width.")
    group.add_argument("--cond_mask_prob", default=0.1, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")

    group.add_argument("--lambda_main", default=1.0, type=float, help="Standard diffusion main objective (just L2 between target and model_output.")
    group.add_argument("--lambda_vel", default=0.0, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_rcxyz", default=0.0, type=float, help="Joint (FK) positions loss.")
    group.add_argument("--lambda_fc", default=0.0, type=float, help="Foot contact loss.")
    group.add_argument("--use_film", action='store_true',
                       help="whether using the FiLM conditioning layer inside the decoder block or not")
    group.add_argument("--target_seq_len", default=150, type=int,
                       help="Target fixed sequence length for batching. Padding if the actual sequence is shorter or sampling the subsequence if longer")
    group.add_argument("--max_persons", default=5, type=int,
                       help="Maximum number of person to perform padding or sampling.  Also used for embedding token")

    group.add_argument("--music_extra_token", action='store_true',
                       help="Whether concat an extra learnable token to represent music or take the (masked) mean along the time")

    # GROUP-STYLE OPTIONS
    group.add_argument("--use_style", action='store_true',
                       help="Whether to use group_embedding and GroupModulation or not")

    group.add_argument("--fuse_style_music", type=str, default='add', choices=['add', 'concat'],
                       help="Whether to fuse the random style_noise and music_repr as input to the mappingnetwork ")

    # GroupDiscriminator options
    group.add_argument("--lambda_contrastive", default=0.0, type=float,
                       help="Using contrastive loss and discriminator")
    group.add_argument("--disc_num_layers", default=3, type=int,
                       help="")
    group.add_argument("--disc_time_aware", action='store_true',
                       help="Add the embed timesteps information of each individual in the positive/negative samples or not")



def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='gdance', choices=['aist', 'gdance'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--num_workers", default=0, type=int,
                       help="Number of workers in DataLoader.")

    group.add_argument("--datapath", default="datasets/gdance_batch_04/", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--split_file", default="split_sequence_names.txt", type=str,
                       help="a split.txt file containing sequences names that will be used")

    group.add_argument("--use_normalizer", action="store_true", help="whether to normalize (scale) data before training or not")


def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--train_platform_type", default='NoPlatform', choices=['NoPlatform', 'ClearmlPlatform', 'TensorboardPlatform'], type=str,
                       help="Choose platform to log results. NoPlatform means no logging.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")


    group.add_argument("--optim", default='Adan',
                       choices=['AdamW', 'Adan'], type=str,
                       help="Architecture types as reported in the paper.")


    # Log and checkpoint options
    group.add_argument("--log_interval", default=50, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=20_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of iterations.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")

    group.add_argument("--pretrained", default="", type=str,
                       help="If not empty, will load the from the pretrained (maybe single-dance) (path to model###.pt file).")

    #wandb
    group.add_argument("--exp_name", default="exp", help="save to project/name")
    group.add_argument(
        "--wandb_pj_name", type=str, default=None, help="project name"
    )

    #contrastive options:
    group.add_argument("--num_negs", default=4, type=int, help="number of negative samples")
    group.add_argument("--neg_sampling_mode", default="mixed_intra_inter", type=str, help="[intra_replace, inter_sample, mixed_intra_inter]")
    group.add_argument("--intra_replace_prob", default=0.5, type=float, help="probability to replace a dancer by another in negative sampling")


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=10, type=int,
                       help="Maximal number of prompts to sampling, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=3, type=int,
                       help="Number of repetitions, per sampling (text prompt/action)")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--timestep_respacing", default="", type=str, help="number of ddim sampling step, for example ddim50")




    group.add_argument("--diversity_scale", default=0.0, type=float,
                       help="diversity scale param")
    group.add_argument("--consistency_scale", default=0.0, type=float,
                       help="discriminator scale param")

    group.add_argument("--fix_n_persons", default=0, type=int,
                       help="make the model generate with desired n_persons")
    group.add_argument("--test_target_seq_len", default=240, type=int, help="the target number of frames to generate")



def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    # Old (legacy) args, dont' use
    group.add_argument("--motion_length", default=6.0, type=float,
                       help="The length of the sampled motion [in seconds]. "
                            "Maximum is 9.8 for HumanML3D (text-to-motion), and 2.0 for HumanAct12 (action-to-motion)")
    group.add_argument("--input_text", default='', type=str,
                       help="Path to a text file lists text prompts to be synthesized. If empty, will take text prompts from dataset.")
    group.add_argument("--action_file", default='', type=str,
                       help="Path to a text file that lists names of actions to be synthesized. Names must be a subset of dataset/uestc/info/action_classes.txt if sampling from uestc, "
                            "or a subset of [warm_up,walk,run,jump,drink,lift_dumbbell,sit,eat,turn steering wheel,phone,boxing,throw] if sampling from humanact12. "
                            "If no file is specified, will take action names from dataset.")
    group.add_argument("--text_prompt", default='', type=str,
                       help="A text prompt to be generated. If empty, will take text prompts from dataset.")
    group.add_argument("--action_name", default='', type=str,
                       help="An action name to be generated. If empty, will take text prompts from dataset.")

    # New args
    group.add_argument("--ema", action='store_true')

    group.add_argument("--test_dataset", default="gdance", type=str, choices=['aist','gdance'])
    group.add_argument("--test_datapath", default="datasets/gdance_batch_04/", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--test_split_file", default="", type=str,
                       help="same as split_file in training_options but for testing")

    group.add_argument("--visualize", action='store_true')
    group.add_argument("--save_result_every", action='store_true', help = "save each resulted motion into separate file")





def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser)


