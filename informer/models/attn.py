import torch
import torch.nn as nn
import math
from math import sqrt
import numpy as np
from utils.masking import TriangularCausalMask, ProbMask

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -1e9)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): 
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) 
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find the Top_k query with the largest sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the Top_k queries to calculate attention
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] 
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) 

        return Q_K, M_top

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() 
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, U_part, u) 

        scale = self.scale or 1. / sqrt(D)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = ProbMask(B, H, L_Q, index, scores_top, device=queries.device)
            scores_top.masked_fill_(attn_mask.mask, -1e9)

        attn = self.dropout(torch.softmax(scale * scores_top, dim=-1))  
        
        context_in_q = torch.matmul(attn, values)
        
        # Map back to the full query length
        context = torch.zeros(B, H, L_Q, D).to(queries.device)
        context[torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :] = context_in_q
        
        context = context.permute(0, 2, 1, 3).contiguous() 

        if self.output_attention:
            return (context, attn)
        else:
            return (context, None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            q, k, v, attn_mask
        )
        out = out.view(B, L, -1)
        return self.out_projection(out), attn