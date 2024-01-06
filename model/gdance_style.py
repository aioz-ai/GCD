import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from model.rotary_embedding_torch import RotaryEmbedding
from model.block import PositionalEncoding, SinusoidalPosEmb, \
    masked_mean, TransformerEncoderLayer, FiLMTransformerDecoderLayer, MappingNetwork, GroupModulation, GlobalAttentionBlock
from model.smpl_skeleton import SMPLSkeleton
from model.discriminator import GroupDiscriminator


class TransformerEncoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, src, mask = None, src_key_padding_mask = None):
        for layer in self.stack:
            src = layer(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return src



"""
USING ONLY GLOBAL ATTENTION AMONG ALL DANCERS
"""
class GDanceStyle(nn.Module):
    # note: this is my implementation of EDGE model,
    # the diagram (fig.2 in the paper) are somewhat confusing at the FiLM layer
    # here i try (mean) pooling the encoded music tokens and add it with the timestep embedding, then utilize them to predict the affine parameters (gammas,betas) of the FiLM layer

    def __init__(self, nfeats, num_actions,
                 normalizer = None,
                 translation=True, pose_rep='rot6d', glob=True,
                 latent_dim=512, ff_size=1024, num_music_enc_layers = 2, num_layers=3, num_heads=4, dropout=0.1,
                 activation="gelu", data_rep='rot6d', dataset='gdance', max_persons = 7, max_seq_len = 150,
                 arch='trans_dec',
                 use_rotary=True,
                 music_extra_token=True, use_film = True,
                 use_style=True, fuse_style_music="add",
                 **kargs):
        super().__init__()

        # pose rep is always rot6d



        self.nfeats = nfeats
        self.num_actions = num_actions # May adapt this to generation with dance style
        self.max_persons = max_persons
        self.max_seq_len = max_seq_len
        self.data_rep = data_rep
        self.dataset = dataset


        self.pose_rep = pose_rep
        self.glob = glob # consider the root orient or not
        self.translation = translation

        self.latent_dim = latent_dim

        self.ff_size = ff_size # Feedforward NN dim
        self.num_music_enc_layers = num_music_enc_layers
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation


        self.normalizer = normalizer

        self.cond_mode = kargs.get('cond_mode', 'no_cond')
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch

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

        #linear projection of the input (data_rep) vectors onto the model dimensionality
        self.input_process = nn.Linear(self.nfeats, self.latent_dim)

        # precomputed PE table has shape (max_len:5000 , 1 , latent_dim:512)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout) #the  input dimension should be (seq_len, batch_size, d)

        self.music_extra_token = music_extra_token # whether append an extra learnable token to represent the whole music sequence

        self.use_film = use_film
        self.use_style = use_style



        # embed the diffusion timestep onto the model to make predictions by first extracting the PE on the timestep and then applying a sequence of linear layers
        # time embedding processing
        self.time_mlp_hidden = nn.Sequential(
            SinusoidalPosEmb(latent_dim),  # learned?
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        self.hidden_to_time_emb = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim), )
        self.hidden_to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),  # 2 time tokens
            Rearrange("b (r d) -> b r d", r=2), #split layer
        )

        # null embeddings for guidance dropout
        self.null_cond_emb = nn.Parameter(torch.randn(1, latent_dim))
        self.null_cond_tokens = nn.Parameter(torch.randn(1, max_seq_len, latent_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # condition processing
        if self.cond_mode != 'no_cond':
            if 'music' in self.cond_mode:
                print('INIT MUSIC TRANSFORMER ENCODER')
                self.inp_music_projection = nn.Linear(4800, self.latent_dim)


                music_encoder_stack = nn.ModuleList([TransformerEncoderLayer(
                            d_model=self.latent_dim,
                            nhead=self.num_heads,
                            dim_feedforward=self.ff_size,
                            dropout=self.dropout,
                            activation=self.activation,
                            batch_first=True,
                            rotary=self.rotary)  for l in range(self.num_music_enc_layers)])
                self.MusicEncoder = TransformerEncoderLayerStack(music_encoder_stack)

                if self.music_extra_token:
                    self._extra_token = nn.Parameter(torch.randn(1, self.latent_dim))

            self.non_attn_cond_projection = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )



        # =============== INIT ARCHITECTURE DECODER LAYERS ======================
        if self.arch == 'trans_dec':
            self.embed_num_persons = EmbedAction(self.max_persons,
                                                 self.latent_dim)  # learnable tokens indicating the desired number of persons to generate
            self.embed_num_persons_mlp = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
            )

            print("TRANS_DEC init")

            if self.use_film:
                print('Using FiLM layers to condition on Transformer Decoder')
                # (Single) Local Attention EDGE's blocks
                self.transformer_decoder_layers = nn.ModuleList([FiLMTransformerDecoderLayer(d_model=self.latent_dim,
                                                                                             nhead=self.num_heads,
                                                                                             dim_feedforward=self.ff_size,
                                                                                             dropout=self.dropout,
                                                                                             activation=activation,
                                                                                             batch_first=True,
                                                                                             rotary=self.rotary) for _ in range(self.num_layers)])
            else:
                print('Using normal Transformer Decoder')
                self.transformer_decoder_layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                                                             nhead=self.num_heads,
                                                                                             dim_feedforward=self.ff_size,
                                                                                             dropout=self.dropout,
                                                                                             activation=activation,
                                                                                             batch_first=True) for _ in range(self.num_layers)])

            # just multihead attention with skip connection
            self.global_attention_layers = nn.ModuleList([GlobalAttentionBlock(embed_dim=self.latent_dim,
                                                                               num_heads=self.num_heads,
                                                                               dropout=self.dropout,
                                                                               batch_first=True,

                                                                               feedforward=True, dim_feedforward=self.ff_size, activation = activation
                                                                               ) for _ in range(self.num_layers)])


        else:
            raise ValueError('Please choose correct architecture currently only support [trans_dec]')

        # GROUP EMBEDDING LAYER
        if self.use_style:
            self.fuse_style_music = fuse_style_music
            if fuse_style_music == "add":
                self.mapping_net = MappingNetwork(input_dim=self.latent_dim, emb_dim=self.latent_dim, depth=8)
            elif fuse_style_music == "concat":
                self.mapping_net = MappingNetwork(input_dim=self.latent_dim*2, emb_dim=self.latent_dim, depth=8)

            self.group_modulation_layers = nn.ModuleList([GroupModulation(self.latent_dim, self.latent_dim) for l in range(self.num_layers)])



        # project the final latent onto the data_representation space (dimension=input_feats),
        # then manipulating the tensor shape to be suitable to the diffusion process (bs, n_joints, n_feats, seq_len)
        self.output_process = nn.Linear(latent_dim, nfeats)

        #self.rot2xyz = Rotation2xyz(device='cpu', dataset=self.dataset)
        self.rot2xyz = SMPLSkeleton()

        # ================= contrastive discriminator =================
        use_discriminator = kargs.get('use_discriminator', False)
        if use_discriminator:
            disc_num_layers = kargs.get('disc_num_layers', 3)
            disc_time_aware = kargs.get('disc_time_aware', False)
            self.discriminator = GroupDiscriminator(nfeats=self.nfeats-4, #ignore 4 contact value
                                                    normalizer=self.normalizer,
                                                    latent_dim=self.latent_dim, ff_size=self.ff_size, num_layers = disc_num_layers,
                                                    num_heads=self.num_heads, dropout=self.dropout, activation=self.activation,
                                                    max_persons=self.max_persons, max_seq_len=self.max_seq_len,
                                                    arch='trans_enc',
                                                    time_aware= disc_time_aware,
                                                    )


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]


    def load_jukebox(self):
        # Currently we extract and pre-cache the music features offline to alleviate the training burden
        pass

    @torch.no_grad()
    def get_attn_mask(self, key_padding_mask):
        bs, seq_len = key_padding_mask.shape
        attn_mask = torch.bmm(key_padding_mask.unsqueeze(2), key_padding_mask.unsqueeze(1))  # shape (bs, seq_len, seq_len), the mask matrix is the outter product: mask @ mask.T
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)  # repeat to (bs*nhead, seq_len, seq_len) as required by torch.transformer

        attn_mask[torch.eye(attn_mask.shape[-1]).repeat(bs * self.num_heads, 1, 1).bool()] = True  # set the diagonal of the mask to True

        attn_mask = ~(attn_mask.bool())

        return attn_mask





    def mask_cond(self, cond, cond_mask, null_cond):
        """
        cond_mask: (bs,)
        """
        bs = cond_mask.shape[0]
        if cond.ndim == 2: #if the condition is only an embeded vector for each sampling
            cond_mask = cond_mask.reshape(bs, 1)
        elif cond.ndim == 3: # the condition is a sequence of tokens (i.e., music)
            cond_mask = cond_mask.reshape(bs, 1, 1)
        # cond_mask: 1-> use null_cond, 0-> use actual cond
        return torch.where(cond_mask, null_cond, cond)  # replace the whole sequence of the masked sample with the learnable null_token

    def get_cond_mask(self, shape, device, force_mask=False):

        if force_mask:
            cond_mask = torch.ones(shape, device=device).bool()
        elif self.training and self.cond_mask_prob > 0.:
            cond_mask = torch.bernoulli(torch.ones(shape, device=device) * self.cond_mask_prob).bool()  # 1-> use null_cond, 0-> use actual cond
        else:
            cond_mask = torch.zeros(shape, device=device).bool()

        return cond_mask





    def forward(self, x, timesteps, y=None, return_style=False):
        """
        x: [bs,  max_persons, nframes, n_feats], denoted x_t in the paper
        timesteps: [batch_size] (int)

        some notes:
         - This architecture only injects the time and cond (text) embed as the input layer to the transformer
            (by concat emb and x  in the first (temporal) dimension

        """
        bs,  max_persons, nframes, n_feats = x.shape

        # ================== Process input music condition ==================
        force_mask = y.get('uncond', False) # 'uncond' means computing null condition for classifier-free-guidance
        if 'music' in self.cond_mode:
            cond_mask = self.get_cond_mask((bs,), device=x.device, force_mask=force_mask)

            inp_music = self.inp_music_projection(y['music']) # linearly project jukebox music features (4800 -> latent_dim)
            src_key_padding_mask = y['frame_mask'].float()  # mask (bs, seq_len)

            if self.music_extra_token:
                extra_token = self._extra_token.unsqueeze(0).expand(inp_music.shape[0], 1, -1)
                inp_music = torch.cat([inp_music, extra_token], dim=1) # (bs, seq_len + 1, latent_dim)
                # extend mask to make it attend to the extra token
                src_key_padding_mask = torch.cat([src_key_padding_mask, torch.ones(bs, 1).to(src_key_padding_mask)], dim=1)

            inp_music = self.abs_pos_encoding(inp_music)

            # src_mask = self.get_attn_mask(src_key_padding_mask)

            emb_music_seq = self.MusicEncoder(src=inp_music,
                                              src_key_padding_mask=~(src_key_padding_mask.bool()),
                                              # mask=src_mask
                                              )



            if self.music_extra_token:
                emb_music_seq, emb_music_extra_token = emb_music_seq[:, :-1, : ], emb_music_seq[:, -1, :]
                emb_music_repr = self.non_attn_cond_projection(emb_music_extra_token) #shape (bs, latent_dim)
            else:
                emb_music_repr = masked_mean(emb_music_seq, mask = src_key_padding_mask.unsqueeze(-1).expand_as(emb_music_seq), dim=1, keepdim=True)
                emb_music_repr = emb_music_repr.squeeze(1)
                emb_music_repr = self.non_attn_cond_projection(emb_music_repr) #shape (bs, latent_dim)

            # in training_mode we have some probability (typically 0.1) to mask the condition by replacing with null_token
            null_cond_tokens = self.null_cond_tokens.to(emb_music_seq)
            emb_music_seq = self.mask_cond(emb_music_seq, cond_mask, null_cond_tokens)
            null_cond_emb = self.null_cond_emb.to(emb_music_repr)
            emb_music_repr = self.mask_cond(emb_music_repr, cond_mask, null_cond_emb)



        # ================== Get timestep embedding (for music condition) ==================
        time_hidden = self.time_mlp_hidden(timesteps) #apply PE and MLP to obtain the hidden time (bs, latent_dim * 4)
        emb_timestep = self.hidden_to_time_emb(time_hidden) # t used in film layer
        emb_time_tokens = self.hidden_to_time_tokens(time_hidden) # (bs,2,latent_dim); t_tokens used for concat with the music sequence for cross-attention






        # ================== Get embedding context ================
        if self.use_film:
            #film gen
            emb_context = emb_timestep
            emb_context +=  emb_music_repr #add the music single representation and the embedding timestep, shape ( bs, latent_dim)

        # also append the embed timestep tokens to perform cross-attention in the decoder layer
        emb_music_seq = torch.cat([emb_music_seq, emb_time_tokens], dim=1)
        src_key_padding_mask = torch.cat([src_key_padding_mask,
                                          torch.ones(bs, emb_music_seq.shape[1]-src_key_padding_mask.shape[1]).to(src_key_padding_mask) ], dim = 1) #(bs, nframes + n_time_tokens)
        emb_music_seq = self.norm_cond(emb_music_seq)

        # ================== mapping to group_embedding ==================
        style_noise = y.get('style_noise', None)  # shape (bs, latent_dim)
        group_embed = None
        if self.use_style and style_noise is not None:
            if self.fuse_style_music == "add":
                style_noise = style_noise + emb_music_repr  # (bs, latent_dim)
            elif self.fuse_style_music == "concat":
                style_noise = torch.cat([style_noise, emb_music_repr], dim=-1)  # (bs, latent_dim*2)
            group_embed = self.mapping_net(style_noise)  # (bs,latent_dim)


        #  ================== Process the motion sequence ==================
        emb_n_persons = self.embed_num_persons(y['n_persons'].unsqueeze(1) - 1)  # -1 because the idx start at 0, (bs, latent_dim)
        emb_n_persons = self.embed_num_persons_mlp(emb_n_persons)

        x = self.input_process(x)

        if self.arch == 'trans_dec':
            xseq = x
            # reshape to correct the positional encoding with proper time frames
            xseq = xseq.view(bs*max_persons, nframes, self.latent_dim)
            xseq = self.abs_pos_encoding(xseq)
            output = xseq

            # Loop through each layer
            for l in range(self.num_layers):
                # ======== Local Attention for each dancer ========
                transformer_decoder_layer = self.transformer_decoder_layers[l]
                # viewing each person as a (local) separate element in the whole larger batch (
                tgt_key_padding_mask = y['data_mask'].reshape(-1,nframes).bool() # an example mask for a sample should look like this [1100 1100 1100 0000] for 3 actual persons and 1 padded person, 2 frames are padded
                tgt_key_padding_mask[:,0] = True # hacky way to handle the tgt_key_padding_mask to avoid nan on padded person, the frame_mask can never be all False

                # repeat for every person of each sample, (bs*max_persons, latent_dim)
                emb_context_repeat = torch.repeat_interleave(emb_context, max_persons, dim=0)
                emb_music_seq_repeat = torch.repeat_interleave(emb_music_seq, max_persons, dim=0)
                src_key_padding_mask_repeat = torch.repeat_interleave(src_key_padding_mask, max_persons, dim=0)

                output = transformer_decoder_layer(tgt=output, memory=emb_music_seq_repeat, t=emb_context_repeat, #  also perform cross-attention of the tgt and the memory (Query is the tgt while Key/Value is the memory)
                                                   tgt_key_padding_mask = ~(tgt_key_padding_mask.bool()),
                                                   memory_key_padding_mask = ~(src_key_padding_mask_repeat.bool()),
                                                   ) # (bs * max_persons, nframes, latent_dim)

                # ======== Global Attention  ========
                output = output.view(bs, max_persons*nframes, self.latent_dim)

                global_attn_layer = self.global_attention_layers[l] # it's just a multi-head self attention with proper mask

                # cat the learnable embedding of num_persons
                output = torch.cat([output, emb_n_persons.unsqueeze(1)], dim=1)  # (bs, npersons * nframes + 1, latent_dim)

                if self.use_style and style_noise is not None:
                    output = torch.cat([output, group_embed.unsqueeze(1)], dim=1)

                tgt_key_padding_mask = y['data_mask'].reshape(bs,-1).bool() #shape (bs, npersons * nframes ),
                tgt_key_padding_mask_extra_one = torch.cat([tgt_key_padding_mask, torch.ones(bs,output.shape[1] - tgt_key_padding_mask.shape[1]).to(tgt_key_padding_mask)], dim=1) #cat the extra embed_n_persons_token

                output = global_attn_layer(output, key_padding_mask=~tgt_key_padding_mask_extra_one)[:,:-1]
                output, emb_n_persons = output[:, :max_persons*nframes], output[:,-1]  # [bs, npersons * nframes, latent_dim], [bs, latent_dim]





                # ======== Apply Group Modulation ========
                if self.use_style and style_noise is not None:
                    group_modulation_layer = self.group_modulation_layers[l]
                    output = group_modulation_layer(output, group_embed, max_persons = max_persons, mask=tgt_key_padding_mask, norm_by_group=True) #mask for calculating masked_mean


                output = output.reshape(bs* max_persons, nframes, self.latent_dim) #reshape to feed into the subsequent local block

        else:
            raise NotImplementedError

        output = output.view(bs, max_persons, nframes, self.latent_dim) # recover the input shape
        output = self.output_process(output)  # project from latent_dim to data_rep

        if return_style:
            return output, group_embed
        else:
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

    BS , N, S, D = [10, 3, 4, 512]

    model = GDanceStyle(nfeats=151, num_actions=1, translation=True, pose_rep='rot6d', cond_mode='music',
                        num_heads=8, max_seq_len=S, max_persons=N,
                        cond_mask_prob=0.2, use_film=True, music_extra_token=True, use_style=True,
                        )

    y = {'music': torch.randn(BS, S, 4800),
         'frame_mask' : torch.tensor([1,1,0,0.]).unsqueeze(0).expand(BS,-1),
         'data_mask': torch.tensor([[1,1,0,0.0],[1,1,0,0.0],[0,0,0,0]]).unsqueeze(0).expand(BS,-1,-1),
         'n_persons': torch.tensor([2]).expand(BS),
         'style_noise': torch.zeros(BS, D)}

    y['music'] = torch.ones(BS, S, 4800)
    model.eval()
    output = model(x=torch.randn(BS,N,S,151) * y['data_mask'].unsqueeze(-1), timesteps = torch.LongTensor([1]*10), y=y)


    # y['music'] = torch.ones(BS, S, 4800)
    # model.eval()
    # output = model(x=torch.ones(BS, N, S, 151), timesteps=torch.LongTensor([1] * 10), y=y)


    print(output.shape)