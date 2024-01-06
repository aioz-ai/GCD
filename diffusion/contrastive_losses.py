
import math
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import copy

def negative_sampling(x, t, y,  N_negs = 4, mode=None, intra_replace_prob=0.5):
    """
    x: [bs, max_persons, ...]
    t: [bs,]
    y['n_persons']: [bs, ] (longtensor)
    """
    bs, max_persons, *repr_dims  = x.shape
    device = x.device

    if mode == 'intra_replace': #randomly replace some dancer of the sample with dancers from other groups
        x_flat = torch.flatten(x, start_dim=0, end_dim=1) # [bs*max_persons, ...]
        t_repeat = t.repeat_interleave(max_persons, dim=0)
        # filter the padded dancers
        persons_mask = torch.arange(max_persons, device=device).expand(bs, -1) < y['n_persons'].unsqueeze(-1)
        persons_mask = persons_mask.reshape(-1) # [bs * max_persons, ]

        valid_mask = ~ torch.block_diag(*([torch.ones((max_persons,max_persons), device=device)]*bs) ).bool() # each row[i] is a mask indicating whether a dancers of column[j] can replace dancer at row[i] or not
        valid_mask = valid_mask * persons_mask
        valid_prob = valid_mask.float() / valid_mask.float().sum(dim=1, keepdim=True) # each candidate has equal replacing probs
        valid_prob = valid_prob[::max_persons] # skip duplicated rows
        # print(valid_prob)

        x_negs, t_negs = [], []
        for k in range(N_negs):
            replace_mask = torch.bernoulli(torch.ones(bs * max_persons,device=device) * intra_replace_prob).bool()# [bs*max_persons, ], True means that a dancer should be replaced with other valid dancers
            replace_mask = replace_mask * persons_mask
            replace_idx = torch.LongTensor(np.array(
                [np.random.choice(bs*max_persons, size=max_persons, p=valid_prob[i].cpu().numpy(), replace=True) for i in range(bs)]
            )).reshape(-1) # [bs*max_persons, ]


            t_neg = t_repeat[replace_idx]
            t_neg = torch.where(replace_mask, t_neg,  t_repeat)
            t_neg = t_neg.view([bs, max_persons]).unsqueeze(1) #[bs , 1, max_persons]
            t_negs.append(t_neg)


            x_replaced = x_flat[replace_idx]
            x_neg = torch.where(replace_mask.view([-1] + [1]*len(x_flat.shape[1:])), x_replaced, x_flat) #first broadcast the replace_mask to shape of x_flat
            assert x_neg.shape == x_flat.shape
            x_neg = x_neg.view([bs, 1, max_persons] + repr_dims) #[bs , 1, max_persons, ...]
            x_negs.append(x_neg)
        t_negs = torch.cat(t_negs, dim=1)  # [bs, N_negs, max_persons]
        x_negs = torch.cat(x_negs, dim=1) # [bs, N_negs, max_persons, ...]

        new_y = {'data_mask': y['data_mask'].unsqueeze(1).repeat_interleave(N_negs, dim=1), # [bs, N_negs, max_persons, n_frames]
                 'frame_mask': y['frame_mask'].unsqueeze(1).repeat_interleave(N_negs, dim=1),  # [bs, N_negs, n_frames]
                 'n_persons': y['n_persons'].unsqueeze(1).repeat_interleave(N_negs, dim=1),  # [bs, N_negs]
                }
        return x_negs, t_negs, new_y
    elif mode == 'inter_sample': #replace the whole group with another group (strengthen connection between music and the corresponding group by contrasting)
        valid_mask = ~torch.eye(bs,device=device).bool() # each row[i] is the mask indicating whether group [j] can replace group [i] or not
        valid_prob = valid_mask.float() / valid_mask.float().sum(dim=1, keepdim=True)  # each candidate has equal replacing probs


        x_negs, t_negs = [], []
        new_data_mask, new_frame_mask, new_n_persons = [], [], []
        for k in range(N_negs):
            replace_idx = torch.LongTensor(np.array(
                [np.random.choice(bs, size=1, p=valid_prob[i].cpu().numpy(), replace=True) for i in range(bs)]
            )).reshape(-1)  # [bs, ]


            t_neg = t[replace_idx]
            t_neg = t_neg.view(bs, 1, 1).repeat(1,1,max_persons) #[bs, 1, max_persons]
            t_negs.append(t_neg)

            x_replaced = x[replace_idx] #[bs, max_persons, ...]
            x_neg = x_replaced.view([bs, 1, max_persons] + repr_dims) #[bs , 1, max_persons, ...]
            x_negs.append(x_neg)

            new_data_mask.append(y['data_mask'][replace_idx].unsqueeze(1))
            new_frame_mask.append(y['frame_mask'][replace_idx].unsqueeze(1))
            new_n_persons.append(y['n_persons'][replace_idx].unsqueeze(1))

        t_negs = torch.cat(t_negs, dim=1)  # [bs, N_negs, max_persons]
        x_negs = torch.cat(x_negs, dim=1)  # [bs, N_negs, max_persons, ...]
        new_y = {
            'data_mask': torch.cat(new_data_mask, dim=1), # [bs, N_negs, max_persons, n_frames]
            'frame_mask':torch.cat(new_frame_mask, dim=1), # [bs, N_negs, n_frames]
            'n_persons': torch.cat(new_n_persons, dim=1) # [bs, N_negs]
        }
        return x_negs, t_negs, new_y
    elif mode =='mixed_intra_inter':
        assert N_negs % 2 == 0
        x_negs1, t_negs1, new_y1 = negative_sampling(x, t, y, N_negs=N_negs // 2, mode='intra_replace', intra_replace_prob=intra_replace_prob)
        x_negs2, t_negs2, new_y2 = negative_sampling(x, t, y, N_negs=N_negs // 2, mode='inter_sample', intra_replace_prob=intra_replace_prob)

        t_negs = torch.cat([t_negs1, t_negs2], dim=1)
        x_negs = torch.cat([x_negs1, x_negs2], dim=1)
        new_y = {}
        for key in ['data_mask', 'frame_mask', 'n_persons']:
            new_y[key] = torch.cat([new_y1[key], new_y2[key]],dim=1)
        return x_negs, t_negs, new_y

def contrastive_losses(model, discriminator, x_start, x, style_embed, t, y, diffusion = None,
                       num_negs=4, sampling_mode='mixed_intra_inter', intra_replace_prob=0.5,
                       re_denoise = False):
    """
    x: [bs, max_persons, nframes, n_feats], groundtruth or predicted group sequences
    t: [bs, ] (longtensor), diffusion timestep
    style_embed: [bs, latent_dim], group representation
    y: dict, model_kwargs
    """

    bs, max_persons, *repr_dims = x.shape
    if bs <= 1:
        return torch.zeros(bs).to(x)



    # forward positive samples
    positive_logit = discriminator(x, style_embed,
                                    timesteps = t.view(bs, 1).repeat(1, max_persons),
                                    y = y,
                                    ) #[bs, 1]

    # negative sampling
    x_negs = []
    if not re_denoise:
        x_negs, t_negs, y_negs = negative_sampling(x, t, y, N_negs=num_negs, mode=sampling_mode, intra_replace_prob=intra_replace_prob)
    else:
        x_start_negs, _, y_negs = negative_sampling(x_start, t, y, N_negs=num_negs, mode=sampling_mode, intra_replace_prob=intra_replace_prob)
        noise = torch.randn_like(x_start_negs)
        x_t_negs = diffusion.q_sample(x_start_negs, t, noise=noise)
        t_negs = t.unsqueeze(1).expand(-1,num_negs)


    # denoise negative samples
    # x_negs = torch.flatten(x_negs, start_dim=0, end_dim=1)  # [bs*max_persons, ...]
    negative_logits = []
    for j in range(num_negs):
        t_neg = t_negs[:, j]
        y_neg = copy.deepcopy(y)
        for key in ['data_mask', 'frame_mask', 'n_persons']:
            y_neg[key] = y_negs[key][:, j]

        if re_denoise:
            x_t_neg = x_t_negs[:, j]
            x_neg, _ = model(x_t_neg, diffusion._scale_timesteps(t_neg), y=y_neg, return_style=True)
            x_neg, _ = torch.split(x_neg, (x_neg.shape[-1] - 4, 4), dim=-1) #strip off contact
            x_negs.append(x_neg.unsqueeze(1))
            t_neg = t_neg.unsqueeze(1).expand(-1, max_persons)
        else:
            x_neg = x_negs[:, j]


        negative_logit = discriminator(x_neg, style_embed,
                                        timesteps = t_neg,
                                        y = y_neg
                                        ) #[bs, 1]
        negative_logits.append(negative_logit)
    negative_logits = torch.cat(negative_logits, dim=1)

    # compute loss
    logits = torch.cat([positive_logit, negative_logits], dim=1) # [bs, 1+num_negs]
    labels = torch.zeros(bs, dtype=torch.long, device=logits.device) # true (positive) logit is at label 0

    loss = F.cross_entropy(logits, labels, reduction='none') #shape [bs, ]

    return loss



















