from model import MainNet, LTE, SearchTransfer

import torch
import torch.nn as nn
import torch.nn.functional as F

class TTSR(nn.Module):
    def __init__(self, args):
        super(TTSR, self).__init__()
        self.args = args
        self.num_res_blocks = list( map(int, args.num_res_blocks.split('+')) )

        self.LTE      = LTE.LTE(requires_grad=True)
        self.LTE_copy = LTE.LTE(requires_grad=False) ### used in transferal perceptual loss

        main_in_channels = 256
        self.reduce_dimensionality = nn.Identity()
        if args.model_gen != "ttsr-raw":
            self.reduce_dimensionality = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
            main_in_channels = 64

        self.MainNet = MainNet.MainNet(in_channels=main_in_channels, num_res_blocks=self.num_res_blocks, n_feats=args.n_feats, res_scale=args.res_scale)

        if args.model_gen in ["ttsr-raw", "ttsr-reduced"]:
            self.SearchTransfer = SearchTransfer.SearchTransfer()
        elif args.model_gen in ["ttsr-soft-attention"]:
            self.SearchTransfer = SearchTransfer.SearchTransferSoftAttn()
        else:
            self.SearchTransfer = SearchTransfer.SearchTransferTrainableWeights()

    def forward(self, lr=None, lrsr=None, ref=None, refsr=None, sr=None, return_attention=False):
        if (type(sr) != type(None)):
            ### used in transferal perceptual loss
            self.LTE_copy.load_state_dict(self.LTE.state_dict())
            sr_lv1, sr_lv2, sr_lv3 = self.LTE_copy((sr + 1.) / 2.)
            return sr_lv1, sr_lv2, sr_lv3

        lrsr_lv1, lrsr_lv2, lrsr_lv3  = self.LTE((lrsr.detach() + 1.) / 2.)
        _, _, refsr_lv3 = self.LTE((refsr.detach() + 1.) / 2.)

        ref_lv1, ref_lv2, ref_lv3 = self.LTE((ref.detach() + 1.) / 2.)

        lrsr_lv3 = self.reduce_dimensionality(lrsr_lv3)
        refsr_lv3 = self.reduce_dimensionality(refsr_lv3)
        ref_lv3 = self.reduce_dimensionality(ref_lv3)

        if not return_attention:
            S, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv1, lrsr_lv2, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3, return_attention)
        else:
            S, S_args, T_lv3, T_lv2, T_lv1 = self.SearchTransfer(lrsr_lv1, lrsr_lv2, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3, return_attention)

        sr = self.MainNet(lr, S, T_lv3, T_lv2, T_lv1)

        if not return_attention:
            return sr, S, T_lv3, T_lv2, T_lv1
        else:
            return sr, S, S_args, T_lv3, T_lv2, T_lv1
