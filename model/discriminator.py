import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from model.rotary_embedding_torch import RotaryEmbedding
from model.block import PositionalEncoding, SinusoidalPosEmb, \
    masked_mean, TransformerEncoderLayer, FiLMTransformerDecoderLayer, MappingNetwork, GroupModulation, GlobalAttentionBlock

class TransformerEncoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, src, mask = None, src_key_padding_mask = None):
        for layer in self.stack:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src

class GroupDiscriminator(nn.Module):
    def __init__(self, nfeats,
                 normalizer = None,
                 pose_rep='rot6d',
                 latent_dim=512, ff_size=1024, num_layers=3, num_heads=4, dropout=0.1, activation="gelu",
                 data_rep='rot6d',
                 dataset='gdance', max_persons = 1, max_seq_len = 150,
                 arch='trans_enc',
                 use_rotary=True,
                 time_aware = False,
                 **kargs):
        super().__init__()

        self.nfeats = nfeats
        self.max_persons = max_persons
        self.data_rep = data_rep
        self.dataset = dataset

        self.latent_dim = latent_dim
        self.ff_size = ff_size # Feedforward NN dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        self.normalizer = normalizer

        self.arch = arch

        self.time_aware = time_aware # the discriminator takes timesteps as input or not

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        # if rotary, replace absolute embedding with a rotary embedding instance (absolute becomes an identity)
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )



        self.input_process = nn.Linear(self.nfeats, self.latent_dim)

        if time_aware:

            self.time_mlp_emb = nn.Sequential(
                SinusoidalPosEmb(latent_dim),  # learned?
                nn.Linear(latent_dim, latent_dim),
                nn.Mish(),
                nn.Linear(latent_dim, latent_dim)
            )

        self.style_mlp_hidden = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim*2),
            nn.SiLU(),
        )
        self.style_mlp_emb_local = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim),
        )
        self.style_mlp_emb = nn.Sequential(
            nn.Linear(latent_dim*2, latent_dim),
        )
        if self.arch == 'trans_enc':
            self.embed_num_persons = EmbedAction(self.max_persons,
                                                 self.latent_dim)  # learnable tokens indicating the desired number of persons to generate
            self.embed_num_persons_mlp = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )
            print("TRANS_ENC init")
            print('Using normal TransformerEncoder')
            self.transformer_encoder_layers = nn.ModuleList([TransformerEncoderLayer(
                                                    d_model=self.latent_dim,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size,
                                                    dropout=self.dropout,
                                                    activation=self.activation,
                                                    batch_first=True,
                                                    rotary=self.rotary)  for _ in range(self.num_layers)])

            self.global_attention_layers = nn.ModuleList([GlobalAttentionBlock(embed_dim=self.latent_dim,
                                                                               num_heads=self.num_heads,
                                                                               dropout=self.dropout,
                                                                               batch_first=True,
                                                                               feedforward=True, dim_feedforward=self.ff_size, activation = activation
                                                                               ) for _ in range(self.num_layers)])

        else:
            raise ValueError('Please choose correct architecture for discriminator, currently only support [trans_enc]')

        self.output_process = nn.Linear(latent_dim, 1)

    def forward(self, x, style_embed, timesteps=None, y=None):
        """
            x: [bs, max_persons, n_frames, n_feats]
            style_embed: [bs, latent_dim]
            timesteps: [bs, max_persons] (longtensor)
            NOTE: the timesteps here are unshared for individual dancers, each persons may have different timestep
            (it is different from the timesteps of the Generator, where dancers of a whole group share the same timestep)
            because the negative sampling may contain dancers from other groups that have different timesteps
        """

        bs, max_persons, nframes, n_feats = x.shape

        style_embed = self.style_mlp_hidden(style_embed) #[bs, latent_dim*2] hidden_dim

        # ================== Get time embedding ==================
        if timesteps is not None and self.time_aware:
            timesteps = timesteps.reshape(-1) # N_all = bs*max_persons
            emb_timesteps = self.time_mlp_emb(timesteps) # [bs*max_persons, latent_dim]
            emb_context_local = emb_timesteps + self.style_mlp_emb_local(style_embed).repeat_interleave(max_persons, dim=0)
            emb_context_local = emb_context_local.unsqueeze(1) #[bs*max_persons, 1, latent_dim]

        emb_context = self.style_mlp_emb(style_embed)
        emb_context = emb_context.unsqueeze(1)


        emb_n_persons = self.embed_num_persons(y['n_persons'].unsqueeze(1) - 1)  # -1 because the idx start at 0, (bs, latent_dim)
        emb_n_persons = self.embed_num_persons_mlp(emb_n_persons)
        emb_n_persons = emb_n_persons.unsqueeze(1) #same shape as the emb_context

        #  ================== Process the motion sequence ==================
        x = self.input_process(x)

        if self.arch == 'trans_enc':
            xseq = x
            xseq = x.view(bs*max_persons, nframes, self.latent_dim)
            xseq = self.abs_pos_encoding(xseq)
            output = xseq


            for l in range(self.num_layers):
                # ========  Apply local attention ========

                transformer_encoder_layer = self.transformer_encoder_layers[l]

                local_tgt_key_padding_mask = y['data_mask'].reshape(-1,nframes).bool() #shape (bs* npersons, nframes), an example mask for a sampling should be like this [1100 1100 1100 0000] for 3 actual persons and 1 padded person, 2 frames are padded
                if timesteps is not None and self.time_aware:
                    output = torch.cat([output, emb_context_local], dim=1)
                    local_tgt_key_padding_mask = torch.cat([local_tgt_key_padding_mask, torch.ones(bs*max_persons, output.shape[1] - local_tgt_key_padding_mask.shape[1]).to(local_tgt_key_padding_mask) ], dim = 1) #cat the extra embed tokens
                else:
                    local_tgt_key_padding_mask[:,0] = True #a hacky way to handle the tgt_key_padding_mask to avoid nan on padded person, the frame_mask can never be all False

                output = transformer_encoder_layer(output,
                                                   src_key_padding_mask =  ~(local_tgt_key_padding_mask.bool())
                                                   )
                if timesteps is not None and self.time_aware:
                    output, emb_context_local = output[:, :-1], output[:,-1:]

                # ======== Global Attention   ========
                output = output.reshape(bs, max_persons * nframes, self.latent_dim)
                global_attn_layer = self.global_attention_layers[l] # it's just a multi-head self attention with proper mask


                output = torch.cat([output, emb_context, emb_n_persons], dim=1) # (bs, npersons * nframes + 1 + 1, latent_dim)
                global_tgt_key_padding_mask = y['data_mask'].reshape(bs,-1).bool() #shape (bs, npersons * nframes ), an example mask should be like this [1100 1100 1100 0000] for 3 actual persons and 1 padded person, 2 frames are padded
                global_tgt_key_padding_mask = torch.cat([global_tgt_key_padding_mask, torch.ones(bs, output.shape[1] - global_tgt_key_padding_mask.shape[1]).to(global_tgt_key_padding_mask) ], dim = 1) #cat the extra embed tokens
                output = global_attn_layer(output,
                                           key_padding_mask=~global_tgt_key_padding_mask)  # (bs, npersons * nframes +1 +1, latent_dim)

                output, emb_context, emb_n_persons  = output[:, :-2], output[:,[-2]], output[:, [-1]] # [bs,max_persons * nframes, latent_dim], [bs, 1, latent_dim]

                output = output.reshape(bs * max_persons, nframes, self.latent_dim)  # reshape to feed into the next local block



        else:
            raise NotImplementedError

        output = output.view(bs, max_persons * nframes, self.latent_dim)
        # the output embedding is the average pooling of the whole sequences
        # output = masked_mean(output, y['data_mask'].reshape(bs,-1,1).expand_as(output),  dim=1) #[bs, latent_dim]
        output = output[:,0] #[bs, latent_dim]

        output = self.output_process(output) #[bs, 1]

        return output







class EmbedAction(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input[:, 0].to(torch.long)  # an index array must be long
        output = self.action_embedding[idx]
        return output


if __name__ == "__main__":
    BS, N, S, D = [10, 3, 4, 512]
    y = {'music': torch.randn(BS, S, 4800),
         'frame_mask' : torch.tensor([1,1,0,0.]).unsqueeze(0).expand(BS,-1),
         'data_mask': torch.tensor([[1,1,0,0.0],[1,1,0,0.0],[0,0,0,0]]).unsqueeze(0).expand(BS,-1,-1),
         'n_persons': torch.tensor([2]).expand(BS),
         'style_noise': torch.zeros(BS, D)}
    model = GroupDiscriminator(nfeats=151, time_aware=True, max_persons=3)
    print(model.transformer_encoder_layers[0].linear1.weight[:2,:5])
    output = model(x=torch.randn(BS, N, S, 151) * y['data_mask'].unsqueeze(-1),
                   style_embed = torch.randn(BS, D),
                   timesteps=torch.LongTensor([1] * (BS*N)).view(BS,N),
                   y=y)
    print(output.shape)





