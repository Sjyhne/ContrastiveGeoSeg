import torch
import torch.nn as nn
import torch.nn.functional as F

import einops
import numpy as np
import math
import time

from tools.cfg import py2cfg
from tools.load_pretrained import load_checkpoint

from .sub_models.class_attention import ClassAttention, ProjectionLayer, LightWeightClassAttention, PosParam, NegParam
from .sub_models.deformable_attention import DAttentionBaseline, TransformerMLPWithConv
from .DCSwin import Decoder
from .CSwin import UniFormer

from scipy.ndimage import distance_transform_edt as distance
from torchvision import transforms
from functools import partial
from operator import itemgetter


def random_patch(class_image, not_class_image, mask, patch_size):
    _, _, h, w = class_image.size()
    if patch_size >= h or patch_size >= w:
        raise ValueError("Patch size must be smaller than the image size")
    x = torch.randint(0, w - patch_size, (1,))
    y = torch.randint(0, h - patch_size, (1,))
    class_patch = class_image[:, :, y:y+patch_size, x:x+patch_size]
    not_class_patch = not_class_image[:, :, y:y+patch_size, x:x+patch_size]
    mask_patch = mask[:, :, y:y+patch_size, x:x+patch_size]
    return class_patch, not_class_patch, mask_patch

def split_tensor(tensor, dim=0):
    return torch.split(tensor, tensor.size(dim) // 2, dim=dim)

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def calculcate_offset_line_length(offset):
    x = offset[:, :, :, 1]
    y = offset[:, :, :, 0]
    
    return torch.sqrt(x**2 + y**2)

def get_dist_maps(target):
        target_detach = target.clone().detach()
        
        dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])
        
        dist_maps = torch.cat([dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

class ContrastiveDCCSwin(nn.Module):
    def __init__(self, config, contrastive_loss) -> None:
        super().__init__()
        
        self.config = config
        self.contrastive_indices = torch.tensor(config["con_conf"]["contrastive_indices"])
        self.use_dist_loss = config["con_conf"]["dist_loss"]
        self.attn_loss_clip = config["con_conf"]["attn_loss_clip"]
        self.con_loss_clip = config["con_conf"]["con_loss_clip"]
        self.reduced_contrastive_patch = config["con_conf"]["reduced_contrastive_patch"]
        self.take_top = config["con_conf"]["take_top"]
        
        self.patch_size = config.patch_size
        
        self.learning_rate = config.learning_rate
        
        self.backbone = UniFormer(**config["cswin_conf"])
        
        self.loss_type = config["con_conf"]["loss_type"]
        
        load_checkpoint(self.backbone, "pretrain_weights/uniformer.pth")
        
        self.decoder = Decoder(**config["decoder_conf"])

        self.projection = nn.ModuleList(
                [ProjectionLayer(
                    in_dim, 
                    config["proj_conf"]["out_features"]
                ) for in_dim in config["proj_conf"]["in_features"]
            ])
        
        self.cl = contrastive_loss
        
    def forward(self, x, mask=None, type="train"):
        features = self.backbone(x)
        
        x = self.decoder(*features)

        if mask == None:
            return x

        contrastive_loss = self.contrastive(features=features, mask=mask, type=type)

        return contrastive_loss, x

    def contrastive_loss(self, pos_feats, neg_feats, current_device):
        
        num_pos = pos_feats.size(0)
        num_neg = neg_feats.size(0)
        
        #topk_start_percentage = 0.0
        #topk_end_percentage = topk_start_percentage + 0.4
        
        con_loss = torch.tensor([0.], device=current_device)
            
        #similarity_dim = -1
        
        if num_neg >= 5 and num_pos >= 5:
            
            num_examples = min(num_neg, num_pos)
            
            perm = torch.randperm(pos_feats.size(0))
            pos_feats = pos_feats[perm]
            
            perm = torch.randperm(neg_feats.size(0))
            neg_feats = neg_feats[perm]
            
            con_loss = con_loss + self.cl(pos_feats[:num_examples], neg_feats[:num_examples], 0)

        if num_pos >= 10:
            splitted_pos = split_tensor(pos_feats)
            first_pos_feats, second_pos_feats = splitted_pos[0], splitted_pos[1]
            
            con_loss = con_loss + self.cl(first_pos_feats, second_pos_feats, 1)
        
        return con_loss
        
    def info_nce_loss(self, pos_feats, neg_feats, current_device):
        num_pos = pos_feats.size(0)
        num_neg = neg_feats.size(0)
        
        if num_pos >= 10 and num_neg >= 10:
        
            # Randomize the order of patches
            perm = torch.randperm(pos_feats.size(0))
            pos_feats = pos_feats[perm]
            
            perm = torch.randperm(neg_feats.size(0))
            neg_feats = neg_feats[perm]
            
            max_samples = min(num_pos, num_neg, 2000)
            
            pos_feats = pos_feats[:max_samples]
            neg_feats = neg_feats[:max_samples]
            
            neg_similarity = torch.cosine_similarity(pos_feats, neg_feats, dim=-1)
            
            _, indices = torch.sort(neg_similarity, descending=True)
            
            top_indices = indices[:int(len(indices) * self.take_top)]
            
            pos_feats = pos_feats[top_indices]
            neg_feats = neg_feats[top_indices]
            
            splitted = split_tensor(pos_feats)
            query, positive = splitted[0], splitted[1]
            
            # Need the same number of anchor and query examples, but negative does not need to be the same
            
            loss = self.cl(query, positive, neg_feats)
            
            return loss
        elif num_pos >= 5 and num_neg >= 5:
            perm = torch.randperm(pos_feats.size(0))
            pos_feats = pos_feats[perm]
            
            perm = torch.randperm(neg_feats.size(0))
            neg_feats = neg_feats[perm]
            
            loss = self.cl(pos_feats, torch.flip(pos_feats, dims=[0]), neg_feats)

        return torch.tensor([0.], device=current_device)

    
    def contrastive(self, features, mask, type="train"):
        
        current_device = mask.device
        
        # Choose a random class for each batch
        classes = torch.unique(mask)
        classes_mask = classes != 6
        classes = torch.masked_select(classes, classes_mask)
        con_loss = torch.tensor([0.], device=current_device)
        
        index_masks = [einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=self.patch_size*(2**i), pw=self.patch_size*(2**i)).to(torch.float) for i in self.contrastive_indices]
        
        for i in self.contrastive_indices.to(current_device):
            
            index_mask = index_masks[i]
            feats = features[i]
            
            
            index_mask = einops.rearrange(index_mask, "b c h w -> (b h w) c")
            feats = einops.rearrange(feats, "b c h w -> (b h w) c")
            
            feats = self.projection[i](feats)
            feats = F.normalize(feats, dim=-1)

            for c in classes:
                
                cindex_mask = torch.where(index_mask == c, 1, 0)
                cindex_mask = cindex_mask.to(torch.float32).mean(-1)
                class_features = feats[cindex_mask == 1]
                
                if class_features.size(0) < 5:
                    continue
                
                not_class_features = feats[cindex_mask == 0.]
                
                if not_class_features.size(0) < 5:
                    continue
                
                if self.loss_type == "contrastive":
                    con_loss = con_loss + self.contrastive_loss(class_features, not_class_features, current_device)
                elif self.loss_type == "info_nce":
                    con_loss = con_loss + self.info_nce_loss(class_features, not_class_features, current_device)
                elif self.loss_type == "supcon":
                    con_loss += self.binary_supcon_loss(class_features, not_class_features, current_device)       
                    
        con_loss = con_loss / len(self.contrastive_indices) / len(classes)
            
        return con_loss
    


if __name__ == "__main__":
    
    config = py2cfg("config/potsdam/contrastive_dcswin.py")
    
    model = ContrastiveDCSwin(config)
    