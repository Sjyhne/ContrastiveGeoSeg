import torch
import torch.nn as nn
import einops
from geoseg.models.DCSwin import SwinTransformer

class ClassAttention(nn.Module):
    def __init__(self, num_classes, config):
        super(ClassAttention, self).__init__()
        self.num_classes = num_classes
        self.embed_dims = config["dimensions"]
        self.encodings = [nn.Parameter(torch.empty(1)) for _ in range(2)]
        
        self.embedding_dim = self.embed_dims[0]
        self.depths = config["depths"]
        self.heads = config["heads"]
        self.frozen_stages = config["frozen_stages"]
        
        self.swin = SwinTransformer(embed_dim=self.embedding_dim, depths=self.depths, num_heads=self.heads, frozen_stages=self.frozen_stages, in_chans=1)
        
        old_dict = torch.load("pretrain_weights/stseg_base.pth")['state_dict']
        model_dict = self.swin.state_dict()
        old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        model_dict.update(old_dict)
        self.swin.load_state_dict(model_dict)
        
        for encoding in self.encodings:
            encoding = torch.nn.init.uniform_(encoding)
        
        self.norms = nn.ModuleList([nn.LayerNorm(ec) for ec in self.embed_dims])
        
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
            o = o.contiguous()
            o = self.norms[odx](einops.rearrange(o, 'b c h w -> b h w c'))
            out[odx] = einops.rearrange(o, 'b h w c -> b c h w')
        
        return out


class ProjectionLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ProjectionLayer, self).__init__()
        
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):
        return self.fc(x)