from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor


import math




def masked_mean(x, mask, dim, keepdim=False):
    assert x.shape == mask.shape
    non_mask_len = mask.float().sum(dim=dim, keepdim=True) # number of non masked entry
    non_mask_len = torch.clamp(non_mask_len, min=1) # because we need at least 1 entry to avoid division by zero
    mean = (x*mask).sum(dim=dim, keepdim=True) / non_mask_len

    if keepdim == False:
        mean = mean.squeeze()
    return mean




def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = np.prod(size[1:])

    return math.sqrt(2.0 / fan_in)


class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initWeightsToZero=False,
                 initBiasToZero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initWeightsToZero:
            self.module.weight.data.fill_(0)
        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels, bias=bias), **kwargs)

class MappingNetwork(nn.Module):
    def __init__(self, input_dim, emb_dim, depth=8, lr_mul = 1.0):
        super().__init__()

        layers = []

        # layers.extend([EqualizedLinear(input_dim, emb_dim, lr_mul), leaky_relu()])
        # for i in range(depth-1):
        #     layers.extend([EqualizedLinear(emb_dim, emb_dim, lr_mul), leaky_relu()])

        layers.extend([nn.Linear(input_dim, emb_dim), leaky_relu()])
        for i in range(depth-1):
            layers.extend([nn.Linear(emb_dim, emb_dim), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class GroupModulation(nn.Module):

    def __init__(self, input_dim, output_dim, epsilon=1e-8):
        super(GroupModulation, self).__init__()
        self.epsilon = epsilon
        self.styleModulator = EqualizedLinear(input_dim, 2 * input_dim, equalized=False,
                                              initWeightsToZero=False,
                                              initBiasToZero=True) #if equalized=False then it is just a normal Linear
        # self.styleModulator = nn.Linear(input_dim, 2 * output_dim)
        self.output_dim = output_dim

    def forward(self, x, w, max_persons, mask=None, norm_by_group=True):

        # x: (bs, npersons * nframes,  latent_dim)
        # mask: (bs, npersons * nframes), dtype bool
        bs, seq_len, d = x.size()

        if not norm_by_group: #norm by entity
            if mask is None:
                x = x.view(bs , max_persons, -1,  d)
                mux = x.mean(dim=2, keepdim=True) # (bs, max_persons, 1,  d)
                varx = torch.clamp((x*x).mean(dim=2, keepdim=True) - mux*mux, min=0)
                varx = torch.rsqrt(varx + self.epsilon) #1/sqrt
                x = (x - mux) * varx
                x = x.view(bs, seq_len, d)
            else: # calculate masked mean
                x = x.view(bs , max_persons, -1,  d)
                mask = mask.view(bs, max_persons, -1,  1).expand_as(x)
                mux = masked_mean(x, mask, dim=2, keepdim=True)  #  (bs, max_persons, 1,  d)
                varx = torch.clamp(masked_mean(x*x, mask, dim=2, keepdim=True) - mux*mux,min=0)
                varx = torch.rsqrt(varx + self.epsilon)
                x = (x - mux) * varx
                x = x.view(bs, seq_len, d)
        else: #normalize across the entire group sampling (both frames and persons)
            if mask is None:
                mux = x.mean(dim=1, keepdim=True) # (bs,1, d)
                varx = torch.clamp((x*x).mean(dim=1, keepdim=True) - mux*mux, min=0)
                varx = torch.rsqrt(varx + self.epsilon) #1/sqrt
                x = (x - mux) * varx
            else:
                mask = mask.view(bs, seq_len, 1).expand_as(x)
                mux = masked_mean(x, mask, dim=1, keepdim=True)
                varx = torch.clamp(masked_mean(x*x, mask, dim=1, keepdim=True) - mux*mux,min=0)
                varx = torch.rsqrt(varx + self.epsilon)
                x = (x - mux) * varx


        # Adapt style
        w_style = self.styleModulator(w) # (bs, 2*d)
        w_A, w_B = torch.split(w_style, self.output_dim, dim=-1)

        w_A = w_A.view(bs,1, d)
        w_B = w_B.view(bs,1, d)

        return w_A * x + w_B
        # return (w_A + 1) * x + w_B


# Similar to TransformerEncoderLayer
class GlobalAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,  dropout=0. ,batch_first=False,
                 feedforward = False, dim_feedforward = 1024, activation="relu",
                 norm_first=True):

        super(GlobalAttentionBlock, self).__init__()
        self.embed_dim = embed_dim

        self.norm_first = norm_first

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.feedforward = feedforward
        if self.feedforward:
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = _get_activation_fn(activation)



    def forward(self, tgt, attn_mask = None, key_padding_mask = None):

        if self.norm_first:
            tgt = tgt + self._sa_block(self.norm1(tgt), attn_mask, key_padding_mask)
            if self.feedforward:
                tgt = tgt + self._ff_block(self.norm2(tgt))
        else:
            tgt = self.norm1(tgt + self._sa_block(tgt, attn_mask, key_padding_mask))
            if self.feedforward:
                tgt = self.norm2(tgt + self._ff_block(tgt))

        return tgt

    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        # qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        qk = x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)




# =====================================================================================================

# absolute positional embedding used for vanilla transformer sequential data
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500, batch_first=False):
        super().__init__()
        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):

        if self.batch_first:
            """
            x should have shape (bs, seq_len, d)
            """
            x = x + self.pe.permute(1, 0, 2)[:, : x.shape[1], :]
        else:
            """
            x should have shape (seq_len, bs, d)
            """
            x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


# very similar positional embedding used for diffusion timesteps
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb





# =====================================================================================================

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



class FiLMTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        self.rotary = rotary
        self.use_rotary = rotary is not None

    # x, cond, t
    def forward(
        self,
        tgt,
        memory,
        t,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # self-attention -> film -> residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + featurewise_affine(x_1, self.film1(t))
            # cross-attention -> film -> residual
            x_2 = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask
            )
            x = x + featurewise_affine(x_2, self.film2(t))
            # feedforward -> film -> residual
            x_3 = self._ff_block(self.norm3(x))
            x = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x
                + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x
                + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask),
                    self.film2(t),
                )
            )
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # self-attention block
    # qkv
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    # qkv
    def _mha_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem
        x = self.multihead_attn(
            q,
            k,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))