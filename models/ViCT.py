import math

import torch
import torch.nn as nn

from einops import rearrange, repeat
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim , mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, num, embed_dim, num_heads, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension needs to be divisible by number of heads."

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        self.spatial_attn = nn.Sequential(
            nn.Conv2d(num_heads, num_heads, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.spectral_attn = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim//2),
            nn.GELU(),
            nn.Linear(self.head_dim//2, self.head_dim),
            nn.Dropout(dropout)
        )

        self.attn = None

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y=None):
        B, N, _ = x.shape
        if self.num_heads >1:
            qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            # 计算空间注意力（沿特征维度）
            max_pool = v.max(dim=-1, keepdim=True)[0]  # [B, H, N, 1]
            avg_pool = v.mean(dim=-1, keepdim=True)  # [B, H, N, 1]
            min_pool = v.min(dim=-1, keepdim=True)[0]  # [B, H, N, 1]
            # std_pool = v.std(dim=-1, keepdim=True)  # [B, H, N, 1]
            spatial_pool = torch.cat([max_pool, avg_pool, min_pool], dim=-1)  # [B, H, N, 3]

            # 计算光谱注意力（沿序列维度）
            spa_max = v.max(dim=-2, keepdim=True)[0]  # [B, H, 1, D]
            spa_avg = v.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
            spa_min = v.min(dim=-2, keepdim=True)[0]  # [B, H, 1, D]
            # spa_std = v.std(dim=-2, keepdim=True)  # [B, H, 1, D]
            spectral_pool = torch.cat([spa_max, spa_avg, spa_min], dim=-2)  # [B, H, 3, D]

            # 并行处理所有头
            spatial_attn = F.sigmoid(self.spatial_attn(spatial_pool).sum(dim=-1))  # [B, H, N]
            spectral_attn = F.sigmoid(self.spectral_attn(spectral_pool).sum(dim=-2))  # [B, H, D]

            # 应用联合注意力
            v = v * spatial_attn.unsqueeze(-1) * spectral_attn.unsqueeze(-2)

            self.attn = (spatial_attn.unsqueeze(-1) * spectral_attn.unsqueeze(-2)).transpose(1, 2).reshape(B, N, self.embed_dim)

            attn_self = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_self = F.softmax(attn_self, dim=-1)
            out_self = (attn_self @ v).transpose(1, 2).reshape(B, N, self.embed_dim)
        else:
            qkv = self.qkv_proj(x).reshape(B, N, 3, self.embed_dim).permute(2, 0, 1, 3)
            q, k, v = qkv.unbind(0)
            attn_self = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_self = F.softmax(attn_self, dim=-1)
            out_self = (attn_self @ v).reshape(B, N, self.embed_dim)

        out = self.out_proj(out_self)
        return out


class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 多分支卷积设计
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, math.ceil(in_channels / 4), kernel_size=1),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, math.ceil(in_channels / 4), kernel_size=3, padding=1),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, math.ceil(in_channels / 4), kernel_size=5, padding=2),
            nn.GELU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, math.ceil(in_channels / 4), kernel_size=1),
            nn.GELU()
        )

    def forward(self, x):
        # 并行多尺度特征提取
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)


class Transformer(nn.Module):
    def __init__(self, num, embed_dim, depth, heads, mlp_dim, dropout):
        super().__init__()

        self.depth = depth
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(embed_dim, MultiHeadAttention(num, embed_dim, num_heads=heads, dropout=dropout))),
                Residual(PreNorm(embed_dim, FeedForward(embed_dim, mlp_dim, dropout=dropout)))
            ]))

        self.inception = nn.Sequential(
            InceptionModule(num),
            nn.Dropout(dropout),
            nn.Conv1d(math.ceil(num / 4)*4, num, kernel_size=1),  # 通道数恢复
            nn.GELU()
        )

    def forward(self, x, y):
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x)
            x = self.inception(x)
            x = ff(x)
            if i<self.depth-1:
                x += y
        return x


class ViCT(nn.Module):
    def __init__(self, patch_size, patch_dim, num_classes, embed_dim, depth, heads, mlp_dim, pool='cls',
                dropout=0., emb_dropout=0., mask_style=None):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_patches = patch_size ** 2
        self.patch_dim = patch_dim
        self.embed_dim = embed_dim

        self.aug = nn.Sequential(
            nn.Linear(patch_size*2, patch_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.patch_to_embedding = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.num_patches+1, embed_dim, depth, heads, mlp_dim, dropout)

        self.norm = nn.Identity() if pool=='pooling' else nn.LayerNorm(embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim) if pool=='pooling' else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, mask=None):
        y = x

        x_x = rearrange(x, 'b c (h w) -> b (c h) w', h=self.patch_size)
        x_y = rearrange(x, 'b c (h w) -> b (c w) h', w=self.patch_size)

        x = torch.cat((x_x, x_y), dim=2)
        x = self.aug(x)
        x = rearrange(x, 'b (c h) w -> b c (h w)', h=self.patch_size)

        x = rearrange(x, 'b c p -> b p c')
        x = self.patch_to_embedding(x)  # [b,n,dim]
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        y = rearrange(y, 'b c p -> b p c')
        y = torch.cat((cls_tokens, y), dim=1)  # [b,n+1,dim]
        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, y)

        # classification: using cls_token output
        x = self.norm(x)

        if self.fc_norm is not None:
            t = x[:, 1:, :]
            t = self.fc_norm(t.mean(1))
        else:
            t = x[:, 0]
        x = self.head(t)

        return x


def build_vict(args, **kwargs):
    model = ViCT(
        patch_size=args.patch_size,
        patch_dim=kwargs['band'],
        num_classes=kwargs['num_classes'],
        embed_dim=kwargs['band'],
        depth=kwargs['depth'],
        heads=kwargs['heads'],
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
    )
    return model