import torch
from einops import rearrange, repeat
from torch import nn
import torch.nn.functional as F

from models.ViCT import ViCT


class ViCT_for_SimMIM(ViCT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.mask_style = kwargs['mask_style']

        self.spectral_mask_token = nn.Parameter(torch.zeros(1, 1, self.num_patches))
        self.spatial_mask_token = nn.Parameter(torch.zeros(1, self.patch_dim, 1))

    def forward(self, x, spectral_mask=None, spatial_mask=None):
        y = x

        x_x = rearrange(x, 'b c (h w) -> b (c h) w', h=self.patch_size)
        x_y = rearrange(x, 'b c (h w) -> b (c w) h', w=self.patch_size)
        x = torch.cat([x_x, x_y], dim=-1)
        x = self.aug(x)
        x = rearrange(x, 'b (c h) w -> b c (h w)', h=self.patch_size)

        B, C, L = x.shape
        if self.mask_style in ['spe', 'both']:
            # 光谱掩码
            spectral_mask_tokens = self.spectral_mask_token.expand(B, C, -1)
            w = spectral_mask.unsqueeze(-1).type_as(spectral_mask_tokens) # b, c, 1
            x = x * (1. - w) + spectral_mask_tokens * w
        if self.mask_style in ['spa', 'both']:
            # 空间掩码
            spatial_mask_tokens = self.spatial_mask_token.expand(B, -1, L)
            w = spatial_mask.unsqueeze(1).type_as(spatial_mask_tokens) # b, 1, l
            x = x * (1. - w) + spatial_mask_tokens * w

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
        x = self.norm(x)

        x = x[:, 1:]
        x = x.transpose(1, 2)
        return x


class SimMIM(nn.Module):
    def __init__(self, encoder, mask_style):
        super().__init__()
        self.encoder = encoder
        self.mask_style = mask_style
        self.decoder = nn.Sequential(
            nn.Conv1d(encoder.embed_dim, encoder.patch_dim, kernel_size=1),
        )

    def forward(self, x, spectral_mask, spatial_mask):
        z = self.encoder(x, spectral_mask, spatial_mask) #z:[b,dim,n]
        x_rec = self.decoder(z)
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        # 仅计算掩码区域的损失
        mask = torch.zeros_like(loss_recon)
        if self.mask_style in ['spe', 'both']:
            spectral_mask = spectral_mask.unsqueeze(-1).float()  # [B,C,1]
            mask += spectral_mask
        if self.mask_style in ['spa', 'both']:
            spatial_mask = spatial_mask.unsqueeze(1).float()  # [B,1,L]
            mask += spatial_mask

        loss_mask = (loss_recon * mask).sum() / (mask.sum() + 1e-5)
        return loss_mask


def build_simmim(args, **kwargs):
    encoder = ViCT_for_SimMIM(
        patch_size=args.patch_size,
        patch_dim=kwargs['band'],
        num_classes=kwargs['num_classes'],
        embed_dim=kwargs['band'],
        depth=kwargs['depth'],
        heads=kwargs['heads'],
        mlp_dim=256,
        dropout=0.1,
        emb_dropout=0.1,
        mask_style=args.mask_style
    )
    model = SimMIM(encoder=encoder, mask_style=args.mask_style)
    return model
