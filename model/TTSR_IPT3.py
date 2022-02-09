import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MainNet, LTE

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0, no_norm = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class VisionTransformer(nn.Module):
    def __init__(self, n_feats, kernel_size):
        super(VisionTransformer, self).__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        embedding_dim = n_feats * kernel_size * kernel_size
        num_heads = 2
        hidden_dim = n_feats * kernel_size * kernel_size * 4
        dropout_rate = 0
        no_norm = False
        num_layers = 1

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, no_norm)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

    def forward(self, lr, ref):

        lr_unfold  = F.unfold(lr, kernel_size=self.kernel_size, padding=self.padding)
        ref_unfold = F.unfold(ref, kernel_size=self.kernel_size, padding=self.padding)

        lr_unfold = lr_unfold.permute(2, 0, 1)
        ref_unfold = ref_unfold.permute(2, 0, 1)

        encoded = self.encoder(ref_unfold)
        decoded = self.decoder(lr_unfold, encoded)

        decoded = decoded.permute(1, 2, 0)

        sr = F.fold(decoded, output_size=lr.size()[-2:], kernel_size=self.kernel_size, padding=self.padding)

        return sr

class TTSR_IPT3(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(TTSR_IPT3, self).__init__()
        self.args = args

        n_colors = 3
        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.head_lr = nn.Sequential(
            conv(n_colors, n_feats, kernel_size),
            ResBlock(conv, n_feats, 5, act=act),
            ResBlock(conv, n_feats, 5, act=act)
        )

        ref_channels = n_colors * 16
        self.head_ref = nn.Sequential(
            nn.Conv2d(n_colors, ref_channels, kernel_size=3, stride=4, padding=1),
            conv(ref_channels, n_feats, kernel_size),
            ResBlock(conv, n_feats, 5, act=act),
            ResBlock(conv, n_feats, 5, act=act)
        )

        kernel_size = 3
        self.body = VisionTransformer(n_feats, kernel_size)

        self.tail = nn.Sequential(
            Upsampler(conv, 4, n_feats, bn=False, act=False),
            conv(n_feats, n_colors, kernel_size)
        )

    # LR: 40x40
    # LR_sr: 160x160 (upsampled LR)
    # Ref: 160x160 (true full res)
    # Ref_sr: 160x160 (donwsampled-upsampled Ref)
    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None, return_attention=False):

        lr = self.head_lr(lr)
        ref = self.head_ref(ref)

        res = self.body(lr, ref)
        res += lr

        sr = self.tail(res)

        return sr, None, None, None, None