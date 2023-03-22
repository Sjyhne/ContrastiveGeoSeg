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
from .DCSwin import SwinTransformer

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

class ContrastiveDCSwin(nn.Module):
    def __init__(self, config, contrastive_loss) -> None:
        super().__init__()
        
        self.config = config
        self.contrastive_indices = torch.tensor(config["con_conf"]["contrastive_indices"])
        self.con_loss_clip = config["con_conf"]["con_loss_clip"]
        self.reduced_contrastive_patch = config["con_conf"]["reduced_contrastive_patch"]
        self.take_top = config["con_conf"]["take_top"]
        self.max_samples = config["con_conf"]["max_samples"]
        
        self.patch_size = config["swin_conf"]["patch_size"]
        
        self.backbone = SwinTransformer(**config["swin_conf"])
        
        self.loss_type = config["con_conf"]["loss_type"]
        
        load_checkpoint(self.backbone, "pretrain_weights/swin_base_patch4_window12_384_22k.pth")
        
        self.decoder = Decoder(**config["decoder_conf"])
        
        if config["detach"] != None:
            self.detach = True
        else:
            self.detach = False

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

        contrastive_loss = self.contrastive(features=features, mask=mask, predictions=x)

        return contrastive_loss, x

    def contrastive_loss(self, pos_feats, neg_feats, current_device):
        
        low_pos, high_pos = pos_feats
        
        num_low = low_pos.size(0)
        num_high = high_pos.size(0)
        num_neg = neg_feats.size(0)
        
        #topk_start_percentage = 0.0
        #topk_end_percentage = topk_start_percentage + 0.4
        
        con_loss = torch.tensor([0.], device=current_device)
            
        #similarity_dim = -1
        
        if num_low >= 2 and num_neg >= 2:
            
            num_examples = min(num_neg, num_low)
            
            con_loss = con_loss + self.cl(low_pos[:num_examples], neg_feats[:num_examples], 0)

        if num_low >= 2 and num_high >= 2:
            
            num_examples = min(num_low, num_high)
            
            con_loss = con_loss + self.cl(low_pos[:num_examples], high_pos[:num_examples], 1)
        
        return con_loss
        
    def info_nce_loss(self, pos_feats, neg_feats, current_device):
        
        low_pos, high_pos = pos_feats
        
        num_low = low_pos.size(0)
        num_high = high_pos.size(0)
        num_neg = neg_feats.size(0)
        
        if num_low >= 2 and num_high >= 2  and num_neg >= 2:
        
            max_samples = min(num_low, num_high, num_neg, self.max_samples)
            
            low_pos = low_pos[:max_samples]
            high_pos = high_pos[:max_samples]
            
            neg_feats = neg_feats[:max_samples]
            
            
            loss = self.cl(low_pos, high_pos, neg_feats)
            
            return loss

        return torch.tensor([0.], device=current_device)

    
    def contrastive(self, features, mask, predictions):
        
        predictions = F.softmax(predictions, dim=1)

        current_device = mask.device
        
        # Choose a random class for each batch
        classes = torch.unique(mask)
        classes_mask = classes != 6
        classes = torch.masked_select(classes, classes_mask)
        con_loss = torch.tensor([0.], device=current_device)
        
        index_masks = [einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=self.patch_size*(2**i), pw=self.patch_size*(2**i)).to(torch.float) for i in self.contrastive_indices]
        predictions = [einops.rearrange(predictions, "b c (h ph) (w pw) -> b c (ph pw) h w", ph=self.patch_size*(2**i), pw=self.patch_size*(2**i)).to(torch.float) for i in self.contrastive_indices]
        
        
        for i in self.contrastive_indices.to(current_device):
            
            index_mask = index_masks[i]
            feats = features[i]
            preds = predictions[i]
            
            
            index_mask = einops.rearrange(index_mask, "b c h w -> (b h w) c")
            feats = einops.rearrange(feats, "b c h w -> (b h w) c")
            
            feats = self.projection[i](feats)
            feats = F.normalize(feats, dim=-1)

            for c in classes:
                    
                cpreds = preds[:, c]
                cpreds = einops.rearrange(cpreds, "b c h w -> (b h w) c")
                
                cpreds = cpreds.mean(-1)
                
                cindex_mask = torch.where(index_mask == c, 1, 0)
                cindex_mask = cindex_mask.to(torch.float32).mean(-1)
                
                # We only want cpreds where class distribution is homogenous
                cpreds = cpreds[cindex_mask == 1]
                class_features = feats[cindex_mask == 1]
                
                _, cpred_indices = torch.sort(cpreds)
                
                index = int(len(cpred_indices) * self.take_top)

                
                low_cpred_indices = cpred_indices[:index]
                high_cpred_indices = cpred_indices[-index:]
                
                low_class_features = class_features[low_cpred_indices]
                high_class_features = class_features[high_cpred_indices]
                
                num_low, num_high = low_class_features.size(0), high_class_features.size(0)
                
                if num_low < 2 or num_high < 2:
                    continue
                
                not_class_features = feats[cindex_mask == 0.]

                num_not = not_class_features.size(0)
                
                if num_not < 4:
                    continue
                
                if self.detach:
                    not_class_features = not_class_features.detach()
                
                if self.loss_type == "contrastive":
                    con_loss = con_loss + self.contrastive_loss([low_class_features, high_class_features], not_class_features, current_device)
                elif self.loss_type == "info_nce":
                    con_loss = con_loss + self.info_nce_loss([low_class_features, high_class_features], not_class_features, current_device)
                    
        con_loss = con_loss / len(self.contrastive_indices) / len(classes)
            
        return con_loss
    


if __name__ == "__main__":
    
    config = py2cfg("config/potsdam/contrastive_dcswin.py")
    
    model = ContrastiveDCSwin(config)
    