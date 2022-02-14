import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SearchTransfer(nn.Module):
    def __init__(self):
        super(SearchTransfer, self).__init__()

        self.w_qs = nn.Linear(576, 576, bias=False)
        self.w_ks = nn.Linear(576, 576, bias=False)
        #self.w_vs = nn.Linear(576, 576, bias=False)

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        index = index.view(views).expand(expanse)
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv1, lrsr_lv2, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3, return_attention):

        # Name              | Role                       | Size
        # ----------------------------------------------------------------
        # lrsr_lv3_unfold   | Query                      | (*, 576, 1600)
        # refsr_lv3_unfold  | Key^T                      | (*, 1600, 576)
        # R_lv3             | Query * Key^T              | (*, 1600, 1600)
        # ref_lv3_unfold    | Value 1                    | (*, 576, 1600)
        # ref_lv2_unfold    | Value 2                    | (*, 4608, 1600)
        # ref_lv1_unfold    | Value 3                    | (*, 9216, 1600)
        # T_lv3_unfold      | Attention 1                | (*, 576, 1600)
        # T_lv2_unfold      | Attention 2                | (*, 4608, 1600)
        # T_lv1_unfold      | Attention 3                | (*, 9216, 1600)

        ### search
        lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
        refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

        refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
        lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

        R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
        R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

        ### transfer
        ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
        ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
        ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

        T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
        T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
        T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

        T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
        T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
        T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

        S = R_lv3_star.view(R_lv3_star.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))

        if not return_attention:
            return S, T_lv3, T_lv2, T_lv1

        S_args = R_lv3_star_arg.view(R_lv3_star_arg.size(0), 1, lrsr_lv3.size(2), lrsr_lv3.size(3))
        return S, S_args, T_lv3, T_lv2, T_lv1

    # def forward(self, lrsr_lv1, lrsr_lv2, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3, return_attention):

    #     # Name              | Role                       | Size
    #     # ----------------------------------------------------------------
    #     # lrsr_lv3_unfold   | Query                      | (*, 576, 1600)
    #     # refsr_lv3_unfold  | Key^T                      | (*, 1600, 576)
    #     # R_lv3             | Query * Key^T              | (*, 1600, 1600)
    #     # ref_lv3_unfold    | Value 1                    | (*, 576, 1600)
    #     # ref_lv2_unfold    | Value 2                    | (*, 4608, 1600)
    #     # ref_lv1_unfold    | Value 3                    | (*, 9216, 1600)
    #     # T_lv3_unfold      | Attention 1                | (*, 576, 1600)
    #     # T_lv2_unfold      | Attention 2                | (*, 4608, 1600)
    #     # T_lv1_unfold      | Attention 3                | (*, 9216, 1600)

    #     lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
    #     refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)

    #     lrsr_lv3_unfold = self.w_qs(lrsr_lv3_unfold.transpose(1, 2)).transpose(1, 2)
    #     refsr_lv3_unfold = self.w_ks(refsr_lv3_unfold.transpose(1, 2)).transpose(1, 2)

    #     # Q*K^T
    #     R_lv3 = torch.matmul(refsr_lv3_unfold.transpose(1, 2), lrsr_lv3_unfold)

    #     # (Q*K^T)/sqrt(d_k)
    #     embedding_dim = lrsr_lv3_unfold.size(1)
    #     R_lv3 *= 1/math.sqrt(embedding_dim)

    #     # softmax((Q*K^T)/sqrt(d_k))
    #     R_lv3 = F.softmax(R_lv3, dim=1)

    #     ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
    #     ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
    #     ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

    #     # Attention = softmax((Q*K^T)/sqrt(d_k))*V
    #     T_lv3_unfold = torch.matmul(ref_lv3_unfold, R_lv3)
    #     T_lv2_unfold = torch.matmul(ref_lv2_unfold, R_lv3)
    #     T_lv1_unfold = torch.matmul(ref_lv1_unfold, R_lv3)

    #     T_lv3 = F.fold(T_lv3_unfold, output_size=lrsr_lv3.size()[-2:], kernel_size=(3,3), padding=1) / (3.*3.)
    #     T_lv2 = F.fold(T_lv2_unfold, output_size=(lrsr_lv3.size(2)*2, lrsr_lv3.size(3)*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
    #     T_lv1 = F.fold(T_lv1_unfold, output_size=(lrsr_lv3.size(2)*4, lrsr_lv3.size(3)*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

    #     if not return_attention:
    #         return None, T_lv3, T_lv2, T_lv1

    #     return None, None, T_lv3, T_lv2, T_lv1