import os
import torch.nn as nn
import torch
import math
import torch.distributed as dist


class Query_model(nn.Module):
    def __init__(self, ft_dim, sd_dim, temperature=1, att_func_type='softmax', pool_type='sum'):
        '''
        ft_dim: feature dim of image patch or text token
        sd_dim: dim of UGT
        temperature: temperature for softmax or sparsemax
        att_func_type: attention normlization function type
        pool_type: pooling type for attention weights
        '''

        super().__init__()

        #activation 
        assert att_func_type in ['softmax', 'sigmoid']
        self.att_func_type = att_func_type

        assert pool_type in ['mean', 'max', 'sum']
        self.pool_type = pool_type

        if self.att_func_type == 'softmax':
            self.att_activation = nn.Softmax(dim=-1)
        else:
            self.att_activation = nn.Sigmoid()

        self.att_dim = sd_dim
        self.temperature = temperature
        
        self.q_map = nn.Sequential(
            nn.LayerNorm(ft_dim),
            nn.Linear(ft_dim, sd_dim),
            nn.GELU(),
            nn.LayerNorm(sd_dim),
            nn.Linear(sd_dim, sd_dim)
        )

    def forward(self, ft, sd, mask=None, return_token_att=False):
        q = self.q_map(ft)

        k = sd
        k = k.unsqueeze(0)
        k = k.transpose(2, 1)
        
        inner_dot = torch.matmul(q, k)

        if return_token_att:
            token_att = inner_dot

        inner_dot = inner_dot / math.sqrt(self.att_dim) #scale dot norm

        if mask is not None:
            
            assert mask.shape == q.shape[:2]
            mask = (mask == 0) * 1

            inner_dot = inner_dot * mask.unsqueeze(-1) #sigmod(-inf) = 0, softmax(-inf) = 0

            if return_token_att:
                token_att = inner_dot

        inner_dot = inner_dot / self.temperature
        att_weight = self.att_activation(inner_dot)

        att_ft = att_weight @ sd

        if self.att_func_type == 'sigmoid':
            att_ft = att_ft / att_weight.sum(dim=-1, keepdim=True)
        
        if return_token_att:
            return token_att, att_ft, sd
        return att_weight, att_ft, sd


class UGT(nn.Module):

    def __init__(self, args, ft_dim, sd_dim, sd_num):
        super().__init__()
        self.space_dict = nn.Parameter(torch.randn(sd_num, sd_dim))
        self.logit_scale = nn.Parameter(torch.ones([1]))

        self.img_query_model = Query_model(ft_dim=ft_dim, sd_dim=sd_dim, temperature=1, att_func_type='softmax', pool_type='sum')
        self.txt_query_model = Query_model(ft_dim=ft_dim, sd_dim=sd_dim, temperature=1, att_func_type='softmax', pool_type='sum')
    
    def forward(self, x_aud, x_vid):
        #calculate UGT-based features
        sd_img_att_weight, sd_img_ft, img_k = self.img_query_model(x_vid.permute(1,0,2), self.space_dict)
        sd_txt_att_weight, sd_txt_ft, txt_k = self.txt_query_model(x_aud.permute(1,0,2), self.space_dict)

        #l2 normalization
        sd_img_ft = sd_img_ft / (sd_img_ft.norm(dim=-1, keepdim=True) + 1e-10)
        sd_txt_ft = sd_txt_ft / (sd_txt_ft.norm(dim=-1, keepdim=True) + 1e-10)

        return sd_img_ft, sd_txt_ft
