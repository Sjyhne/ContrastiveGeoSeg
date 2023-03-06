import torch
import torch.nn as nn
import einops
from .swin_v2 import SwinTransformerV2
from ..DCSwin import SwinTransformer

import torch.nn.functional as F

from timm.models.layers import to_2tuple
import time

import math

class ClassAttentionV2(nn.Module):
    def __init__(self, config):
        super(ClassAttention, self).__init__()
        self.embed_dims = config["embed_dims"]

        self.encodings = [nn.Parameter(torch.empty(1)) for _ in range(2)]
        
        self.swin = SwinTransformerV2(**config["swin_conf"])
        
        self.norms = nn.ModuleList([nn.LayerNorm(ec*2) for ec in self.embed_dims])
        
        for encoding in self.encodings:
            encoding = torch.nn.init.uniform_(encoding)
        
        
    def forward(self, mask, class_idx, negative=False):
        
        unknown = self.encodings[0]
        known = self.encodings[1]
        
        encoding = mask.clone().type(torch.float32)
        
        if negative:
            encoding[mask != class_idx] = known
            encoding[mask == class_idx] = unknown
        else:
            encoding[mask != class_idx] = unknown
            encoding[mask == class_idx] = known
        
        encoding = encoding.unsqueeze(1)
        
        out = self.swin(encoding)
        out = list(out)
        
        for odx, o in enumerate(out):
            #o = einops.rearrange(o, 'b c h w -> b h w c')
            o = self.norms[odx](o)
            D = int(math.sqrt(o.shape[1]))
            out[odx] = einops.rearrange(o, 'b (h w) c -> b h w c', h=D, w=D)
        
        return out

class ClassAttention(nn.Module):
    def __init__(self, config):
        super(ClassAttention, self).__init__()
        self.embed_dims = config["embed_dims"]

        self.encodings = [nn.Parameter(torch.empty(1)) for _ in range(2)]
        
        self.swin = SwinTransformer(**config["swin_conf"])
        
        self.norms = nn.ModuleList([nn.LayerNorm(ec) for ec in self.embed_dims])
        
        for encoding in self.encodings:
            encoding = torch.nn.init.uniform_(encoding)
        
        
    def forward(self, mask, class_idx, negative=False):
        
        s = time.time()
        unknown = self.encodings[0]
        known = self.encodings[1]
        e = time.time()
        print("First part:", e - s)
        
        encoding = mask.clone().type(torch.float32)
        
        if negative:
            encoding[mask != class_idx] = known
            encoding[mask == class_idx] = unknown
        else:
            encoding[mask != class_idx] = unknown
            encoding[mask == class_idx] = known
        
        encoding = encoding.unsqueeze(1)
        
        out = self.swin(encoding)
        out = list(out)
        
        for odx, o in enumerate(out):
            o = einops.rearrange(o, 'b c h w -> b h w c')
            o = self.norms[odx](o)
            out[odx] = o
        
        return out

class PosParam(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.pos_attn = nn.init.uniform_(nn.Parameter(torch.empty(1,), requires_grad=True))

    def forward(self, input):
        return input * self.pos_attn

class NegParam(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.neg_attn = nn.init.uniform_(nn.Parameter(torch.empty(1,), requires_grad=True))
    
    def forward(self, input):
        return input * self.neg_attn

class LightWeightClassAttention(nn.Module):
    def __init__(self, embed_dim, patch_size) -> None:
        super().__init__()
        
        self.embedding_dim = embed_dim
        self.patch_size = patch_size
        self.projection = PatchEmbed(patch_size, in_chans=1, embed_dim=self.embedding_dim, norm_layer=nn.LayerNorm)
        
        
    def forward(self, mask, class_index, pp, np):
        
        pos_attn, neg_attn = pp, np
        
        positive_mask = torch.where(mask == class_index, pos_attn, neg_attn)
        negative_mask = torch.where(mask == class_index, neg_attn, pos_attn)
        
        positive = positive_mask.unsqueeze(1)
        negative = negative_mask.unsqueeze(1)
        
        #positive = einops.rearrange(positive, "b h w c -> b c h w")
        #negative = einops.rearrange(negative, "b h w c -> b c h w")

        proj_neg = self.projection(negative)
        proj_pos = self.projection(positive)
        
        proj_mask = einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=self.patch_size, pw=self.patch_size)
        
        
        return proj_neg, proj_pos, proj_mask
        
class ProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ProjectionLayer, self).__init__()
        
        self.fc1 = nn.Linear(in_features, out_features, bias=bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(out_features, out_features)
        
        
        
    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x

        """
        old_dict = torch.load("pretrain_weights/stseg_base.pth")['state_dict']
        model_dict = self.swin.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        self.swin.load_state_dict(model_dict)
        
        """