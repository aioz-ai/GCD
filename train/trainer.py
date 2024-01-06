import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

from diffusion.resample import LossAwareSampler, UniformSampler
from diffusion.resample import create_named_schedule_sampler
from copy import deepcopy

import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm




def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num_processes): # the appended ".module" in the DDP_model statedict
    return x if num_processes == 1 else wrap(x)


class AccelerateTrainer:
    def __init__(self, args, model, diffusion, train_data, ema_rate=0.9999, wandb_pj_name=None):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        self.num_processes = state.num_processes

        self.args = args

        self.model = model
        self.diffusion = diffusion

        self.train_data = train_data
        self.dataset = args.dataset

        self.cond_mode = model.cond_mode
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.optim_type = args.optim

        self.log_interval = args.log_interval
        self.current_train_step = 0

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type,  diffusion)  # timesteps importance sampling (Eq.18 in the Improved DDPM paper, return the ScheduleSampler object (supporting two types: Uniform and LossSecondMoment)


        # Init EMA model
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.ema_models = [deepcopy(self.model).to(self.accelerator.device) for _ in self.ema_rate]
        for ema_model in self.ema_models:
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)


        self.wandb_pj_name = wandb_pj_name


        self.accelerator.wait_for_everyone()





        self.model = self.accelerator.prepare(self.model)
        optimizer = self.configure_optimizers()
        self.optimizer = self.accelerator.prepare(optimizer)




    def configure_optimizers(self):
        print('Using optimizer:', self.optim_type)
        if self.optim_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optim_type == 'Adan':
            from model.adan import Adan
            optimizer = Adan(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError('optim type is not supported')
        return optimizer



    def load_pretrained(self, checkpoint, num_processes):

        state_dict = checkpoint['state_dict']
        if "pytorch-lightning_version" in checkpoint.keys():
            print("Load from lightning ckpt!")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k in state_dict.keys():
                new_k = k.split(".") #strip the 'model.' in params name
                new_k = ".".join(new_k[1:])
                new_state_dict[new_k] = state_dict[k]
            state_dict = new_state_dict

        self.model.load_state_dict(
            maybe_wrap(state_dict, num_processes), strict=False
        )

    def load_checkpoint(self, checkpoint, num_processes):
        self.model.load_state_dict(
            maybe_wrap(checkpoint["state_dict"], num_processes)
        )

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


        if self.accelerator.is_main_process:
            self.current_train_step = checkpoint['current_train_step']
            print("load checkpoint at step:", self.current_train_step)

            ema_checkpoint = checkpoint['ema_model']
            self.ema_models = [
                self._load_ema_parameters(ema_model, ema_checkpoint, rate) for rate, ema_model in zip(self.ema_rate, self.ema_models)
            ]

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            print("save checkpoint at iteration:", self.current_train_step)
            ckpt = {
                "state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "current_train_step":  self.current_train_step,
            }
            ckpt['ema_model'] = {}
            for ema_rate, ema_model in zip(self.ema_rate, self.ema_models):
                ckpt['ema_model'][f'ema_{ema_rate}'] = {}
                ckpt['ema_model'][f'ema_{ema_rate}']['state_dict'] = ema_model.state_dict()


            torch.save(ckpt, os.path.join(self.args.save_dir, f"step={self.current_train_step}.ckpt"))
            # maintaining the last checkpoint
            torch.save(ckpt, os.path.join(self.args.save_dir, f"last.ckpt"))


            if self.wandb_pj_name:
                print("OVERWRITE last.ckpt to wandb")
                wandb.save(os.path.join(self.args.save_dir, "last.ckpt"), policy='live')



    def _load_ema_parameters(self,ema_model, ema_checkpoint, rate):
        state_dict = ema_checkpoint[f'ema_{rate}']['state_dict'] #state_dict is alread mapped to device
        # ema_params = [state_dict[name].to(self.accelerator.device) for name, _ in self.model.named_parameters()]
        #ema_params = [state_dict[name].to(self.accelerator.device) for name in self.model.state_dict.keys()]
        ema_model.load_state_dict(state_dict)
        return ema_model


    @torch.no_grad()
    def _update_ema(self):
        for rate, ema_model in zip(self.ema_rate, self.ema_models):
            for targ, src in zip(ema_model.parameters(), self.model.parameters()):
                targ.detach().mul_(rate).add_(src, alpha=1 - rate)










    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def training_step(self, batch, batch_idx):

        motion, cond = batch
        bs = motion.shape[0]

        cond['y'].update({'style_noise': torch.randn(bs, self.accelerator.unwrap_model(self.model).latent_dim, device=self.accelerator.device)})

        # if using UniformSampler then weights is a Tensor of full ones with size= (batch_size,)
        t, weights = self.schedule_sampler.sample(batch_size=motion.shape[0],
                                                  device=self.accelerator.device)  # importance sampling the timesteps based on the class-implemented self.weights() function
        # the authors hypothesized that sampling t uniformly causes unnecessary noise in the L_vlb term
        # for LossAwareSampler, timestep with higher loss will have smaller weights (higher w -> smaller p), and weights=1/p (see Eq.18 in the Improved DDPM), es (Figure 2),

        # contrastive loss discriminator

        discriminator = self.accelerator.unwrap_model(self.model).discriminator if (self.args.lambda_contrastive > 0.0) else None
        contrastive_args = {"num_negs": self.args.num_negs,
                            "sampling_mode":self.args.neg_sampling_mode,
                            "intra_replace_prob": self.args.intra_replace_prob}

        # return a losses[dict]
        losses = self.diffusion.training_losses(
            model=self.model,
            x_start=motion,  # [bs, njoints, nfeats , target_seq_len]
            t=t,  # [bs](int) sampled the diffusion timesteps
            model_kwargs=cond,
            dataset=self.train_data.dataset,  # dataset object (HumanML3D, KIT, or AIST..)
            discriminator=discriminator, contrastive_args = contrastive_args
       )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )



        loss = (losses["loss"] * weights).mean()

        return {'loss': loss, "loss_dict": losses}

    def train_loop(self, max_epochs):

        if self.args.resume_checkpoint != "":
            print("Load checkpoint from:", self.args.resume_checkpoint)
            checkpoint = torch.load(
                self.args.resume_checkpoint, map_location=self.accelerator.device
            )
            self.load_checkpoint(checkpoint, self.num_processes)
            del checkpoint
        elif self.args.pretrained != "": #loaded pretrained model can be different in some weights and layers, we will set strict=False
            print("Load pretrained from:", self.args.pretrained)
            checkpoint = torch.load(
                self.args.pretrained, map_location=self.accelerator.device
            )
            self.load_pretrained(checkpoint, self.num_processes)
            del checkpoint






        # load datasets



        # data loaders

        train_dataloader = self.accelerator.prepare(self.train_data)
        # boot up multi-gpu training. test dataloader is only on main process
        progress_bar = (
            partial(tqdm, position=0)
            if self.accelerator.is_main_process
            else lambda x, **kwargs : x
        )

        if self.accelerator.is_main_process:
            if self.wandb_pj_name is not None:
                wandb.init(project=self.wandb_pj_name, name=self.args.exp_name, config=vars(self.args))

            save_dir = self.args.save_dir
            os.makedirs(save_dir, exist_ok=True)

        self.accelerator.wait_for_everyone()


        start_epoch = self.current_train_step // len(train_dataloader)
        for epoch in range(start_epoch, max_epochs):
            # train
            self.train()
            pbar_loop = progress_bar(train_dataloader, desc=f"Epoch {epoch}", miniters=0)
            for batch_idx, batch in enumerate(
                    pbar_loop
            ):

                training_step_outputs = self.training_step(batch, batch_idx)
                loss = training_step_outputs['loss']
                loss_dict = training_step_outputs['loss_dict']


                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()

                # print("DISC weight test: ",self.accelerator.local_process_index, "----",self.accelerator.unwrap_model(self.model).discriminator.transformer_encoder_layers[0].linear1.weight[:2,:5])

                # ema update and train loss update only on main
                if self.accelerator.is_main_process:
                    for key in loss_dict.keys():
                        loss_dict[key] = float(loss_dict[key].mean().detach())

                    loss_dict.update({'step': self.current_train_step+1})
                    pbar_loop.set_postfix(loss_dict)

                    # update every step
                    self._update_ema()

                    if (self.current_train_step % self.log_interval == 0):
                        if self.wandb_pj_name:
                            wandb.log(loss_dict, step=self.current_train_step)

                self.current_train_step += 1

                # Save model
                if (self.current_train_step % self.args.save_interval) == 0:
                    # everyone waits here for the val loop to finish ( don't start next train epoch early)
                    self.accelerator.wait_for_everyone()
                    # save only if on main thread
                    if self.accelerator.is_main_process:
                        self.save_checkpoint()



        if self.accelerator.is_main_process:
            wandb.run.finish()



