# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import pickle as pkl
from utils.fixseed import fixseed
from utils.parser_util import train_args
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion


from prettytable import PrettyTable
import torch
import wandb

from train.trainer import AccelerateTrainer

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params




def main():
    args = train_args()
    #fixseed(args.seed)



    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)


    print("creating data loader...")
    train_dataloader = get_dataset_loader(name=args.dataset, datapath=args.datapath, split_file=args.split_file,
                                          batch_size=args.batch_size, target_seq_len=args.target_seq_len, max_persons = args.max_persons, split="train", num_workers=args.num_workers)
    print("Dataset len:", len(train_dataloader.dataset), ", batch size: ", args.batch_size, ', target_seq_len: ', args.target_seq_len)
    # perform data scaling (normalization) for better training
    if args.use_normalizer:
        print("Using data normalizer")
        dataset = train_dataloader.dataset
        datapath = dataset.datapath
        if not os.path.exists(os.path.join(datapath, "normalizer.pkl")):
            from data_loaders.preprocess import preprocess_data
            preprocess_data(dataset, split="train")
        normalizer = pkl.load(
            open(os.path.join(datapath, "normalizer.pkl"),"rb")
        )
        setattr(train_dataloader.dataset, "normalizer", normalizer)




    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, train_dataloader, model_type=args.model_type)
    # count_parameters(model)

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))


    print("Training...")
    trainer = AccelerateTrainer(args, model, diffusion, train_data=train_dataloader, wandb_pj_name=args.wandb_pj_name)


    max_epochs = args.num_steps // len(train_dataloader) + 1
    trainer.train_loop(max_epochs)




if __name__ == "__main__":
    main()
