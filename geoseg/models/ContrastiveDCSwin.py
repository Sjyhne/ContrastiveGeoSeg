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
from .DCSwin import Decoder, SwinTransformer

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
    def __init__(self, config, attn_loss, contrastive_loss) -> None:
        super().__init__()
        
        self.config = config
        self.contrastive_indices = torch.tensor(config["con_conf"]["contrastive_indices"])
        self.use_dist_loss = config["con_conf"]["dist_loss"]
        self.attn_loss_clip = config["con_conf"]["attn_loss_clip"]
        self.con_loss_clip = config["con_conf"]["con_loss_clip"]
        self.reduced_contrastive_patch = config["con_conf"]["reduced_contrastive_patch"]
        self.take_top = config["con_conf"]["take_top"]
        
        self.patch_size = config["swin_conf"]["patch_size"]
        
        self.backbone = SwinTransformer(**config["swin_conf"])
        
        if config["con_conf"]["binary"] != None:
            self.binary = True
        else:
            self.binary = False
        
        self.loss_type = config["con_conf"]["loss_type"]
        
        load_checkpoint(self.backbone, "pretrain_weights/swin_base_patch4_window12_384_22k.pth")
        
        #old_dict = torch.load("pretrain_weights/stseg_base.pth")['state_dict']
        #model_dict = self.backbone.state_dict()
        #old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        #model_dict.update(old_dict)
        #self.backbone.load_state_dict(model_dict)
        
        self.decoder = Decoder(**config["decoder_conf"])
        
        #old_dict = torch.load("pretrain_weights/stseg_base.pth")['state_dict']
        #model_dict = self.decoder.state_dict()
        #old_dict = {k: v for k, v in old_dict.items() if (k in model_dict)}
        #model_dict.update(old_dict)
        #self.decoder.load_state_dict(model_dict)
        
        class_conf = {
            "embed_dims": config["decoder_conf"]["encoder_channels"],
            "patch_size": self.patch_size
        }
        
        self.projection = nn.ModuleList(
                [ProjectionLayer(
                    in_dim, 
                    config["proj_conf"]["out_features"]
                ) for in_dim in config["proj_conf"]["in_features"]
            ])
        
        if False:
            self.cas = nn.ModuleList([
                LightWeightClassAttention(class_conf["embed_dims"][i], class_conf["patch_size"]*(2**i)) for i in range(len(class_conf["embed_dims"]))
            ])
            
            self.projection = nn.ModuleList(
                [ProjectionLayer(
                    in_dim, 
                    config["proj_conf"]["out_features"]
                ) for in_dim in config["proj_conf"]["in_features"]
            ])
            
            self.attn_projection = nn.ModuleList(
                [ProjectionLayer(
                    in_dim, 
                    config["proj_conf"]["out_features"]
                ) for in_dim in config["proj_conf"]["in_features"]
            ])
        
            self.dattn_projection = nn.ModuleList(
                [ProjectionLayer(
                    in_dim, 
                    config["proj_conf"]["out_features"]
                ) for in_dim in config["proj_conf"]["in_features"]
            ])

            self.layer_norm = nn.ModuleList([
                nn.LayerNorm(dim) for dim in config["norm_conf"]["embedding_dims"]
            ])
                    
            self.dattention = nn.ModuleList([
                DAttentionBaseline(
                    config["dattn_conf"]["feature_map_size"],        
                    config["dattn_conf"]["feature_map_size"],
                    config["dattn_conf"]["heads"][i],
                    config["dattn_conf"]["n_head_channels"][i],
                    config["dattn_conf"]["groups"][i],
                    config["dattn_conf"]["attn_drop"],
                    config["dattn_conf"]["proj_drop"],
                    config["dattn_conf"]["strides"][i],
                    config["dattn_conf"]["offset_range_factor"][i],
                    config["dattn_conf"]["use_pe"],
                    config["dattn_conf"]["dwc_pe"],
                    config["dattn_conf"]["no_off"],
                    config["dattn_conf"]["fixed_pe"],
                    i
                ) for i in range(len(config["dattn_conf"]["heads"]))
            ])
            
            self.upscale = nn.ModuleList(
                [nn.Conv2d(config["norm_conf"]["embedding_dims"][i], config["norm_conf"]["embedding_dims"][i + 1], 
                kernel_size=1, 
                stride=2, 
                padding=0) for i in range(len(config["norm_conf"]["embedding_dims"][:-1]))
                ]
            )
            
            self.pos_param = nn.init.uniform_(nn.Parameter(torch.empty(1), requires_grad=True))
            self.neg_param = nn.init.uniform_(nn.Parameter(torch.empty(1), requires_grad=True))
            
            self.projection_norm = nn.LayerNorm(config["proj_conf"]["out_features"])
        
        self.cl = contrastive_loss
        #self.offset_loss = torch.nn.MSELoss()
        
    def forward(self, x, mask=None, type="train"):
        x1, x2, x3, x4 = self.backbone(x)
        
        features = [x1, x2, x3, x4]

        x = self.decoder(*features)

        if mask == None:
            return x
        
        if self.binary:
            contrastive_loss = self.alt_binary_contrastive(features=features, mask=mask, type=type)
        else:
            contrastive_loss = self.alt_contrastive(features=features, mask=mask, type=type)

        return contrastive_loss, x

    def dist_loss(self, c, mask_patch, pos_offset, neg_offset, current_device):
        # All have the same shape
        pos_mask_patch = mask_patch.clone()
        pos_mask_patch = torch.where(pos_mask_patch == c, 1, 0)
        neg_mask_patch = mask_patch.clone()
        neg_mask_patch = torch.where(neg_mask_patch == c, 0, 1)
        
        H = pos_mask_patch.size(2)
        
        pos_offset_distances = calculcate_offset_line_length(pos_offset)
        neg_offset_distances = calculcate_offset_line_length(neg_offset)
        
        pos_mask_offset_distances = get_dist_maps(pos_mask_patch).to(current_device).tanh().mean(1) / H * 2
        neg_mask_offset_distances = get_dist_maps(neg_mask_patch).to(current_device).tanh().mean(1) / H * 2
        
        pos_loss = self.offset_loss(pos_offset_distances, pos_mask_offset_distances)
        neg_loss = self.offset_loss(neg_offset_distances, neg_mask_offset_distances)
        
        return pos_loss + neg_loss

    def index_loss(self, c, mask_patch, pos_pos, neg_pos, current_device):
        H = pos_pos.size(3)
        
        pos_pos = torch.floor((torch.clip(pos_pos, -1., 1.) + 1.) * H // 2)
        
        pos_mask_patch = mask_patch.clone()
        pos_mask_patch = torch.where(pos_mask_patch == c, 1, 0).to(torch.float).mean(1)
        pos_mask_patch = torch.where(pos_mask_patch > 0., 1, 0)
        
        print("c:", c)
        print("pos_mask_patch:", pos_mask_patch.shape)
        print("pos_pos:", pos_pos.shape)
        print(pos_mask_patch[0, 0, 0])
        print(pos_pos[0, 0, 0, 0])
        print()
        print(pos_mask_patch[0, -1, 0])
        print(pos_pos[0, 0, -1, 0])
        print()
        print(pos_mask_patch[0, 0, -1])
        print(pos_pos[0, 0, 0, -1])
        print()
        print(pos_mask_patch[0, -1, -1])
        print(pos_pos[0, 0, -1, -1])
        print()
        
        pos_pos = torch.unique(pos_pos.flatten(1, -2), dim=-2)
        print("pos_pos:", pos_pos.shape)
        
        # Two steps here
        # 1. Extract the boundary of the masks if we find a good method for it
        # 2. Get the indices of the closest 1 from all zeros, and then set it as the "goal" for the pos
        # Might be wise to scale it according to the range factor?
    
    def binary_contrastive_loss(self, pos_feats, neg_feats, current_device):
        
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
            
            # Negative similarity
            #similarities = F.cosine_similarity(pos_feats[:num_examples], neg_feats[:num_examples], dim=similarity_dim)
            #similarities, _  = torch.sort(similarities, descending=False)
            #similarities = similarities[int(similarities.size(0) * topk_start_percentage):int(similarities.size(0) * topk_end_percentage)]
            #distances = 1 - similarities
            #con_loss += self.cl(distances, label=0)

        if num_pos >= 10:
            splitted_pos = split_tensor(pos_feats)
            first_pos_feats, second_pos_feats = splitted_pos[0], splitted_pos[1]
            
            con_loss = con_loss + self.cl(first_pos_feats, second_pos_feats, 1)
            
            # Positive similarity
            #similarities = F.cosine_similarity(first_pos_feats, second_pos_feats, dim=similarity_dim)
            #similarities, _  = torch.sort(similarities, descending=True)
            #similarities = similarities[int(similarities.size(0) * topk_start_percentage):int(similarities.size(0) * topk_end_percentage)]
            #distances = 1 - similarities
            #con_loss += self.cl(distances, label=1)
        
        #if num_neg >= 10:
        #    splitted_neg = split_tensor(neg_feats)
        #    first_neg_feats, second_neg_feats = splitted_neg[0], splitted_neg[1]
            
        #    con_loss = con_loss + self.cl(first_neg_feats, second_neg_feats, 1)
            
            # Positive similarity
            #similarities = F.cosine_similarity(first_neg_feats, second_neg_feats, dim=similarity_dim)
            #similarities, _  = torch.sort(similarities, descending=True)
            #similarities = similarities[int(similarities.size(0) * topk_start_percentage):int(similarities.size(0) * topk_end_percentage)]
            #distances = 1 - similarities
            #con_loss += self.cl(distances, label=1)
        
        return con_loss
        
    def binary_info_nce_loss(self, pos_feats, neg_feats, current_device):
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

    def binary_supcon_loss(self, pos_feats, neg_feats, current_device):
        num_pos = pos_feats.size(0)
        num_neg = neg_feats.size(0)
        
        # Randomize the order of patches
        perm = torch.randperm(pos_feats.size(0))
        pos_feats = pos_feats[perm]
        
        perm = torch.randperm(neg_feats.size(0))
        neg_feats = neg_feats[perm]
        
        max_samples = 1000
        pos_feats = pos_feats[:max_samples]
        neg_feats = neg_feats[:max_samples]
        
        # Need the same number of anchor and query examples, but negative does not need to be the same
        if num_pos >= 10 and num_neg >= 10:
            #splitted = split_tensor(pos_feats)
            #query, positive = splitted[0], splitted[1]
            
            pos_labels = torch.ones(pos_feats.size(0), device=current_device)
            neg_labels = torch.zeros(neg_feats.size(0), device=current_device)
            labels = torch.cat([pos_labels, neg_labels])
            features = torch.cat([pos_feats, neg_feats])
            features = features.unsqueeze(1)
            
            loss = self.cl(features, labels)
            
            return loss

        return torch.tensor([0.], device=current_device)
    
    def binary_contrastive(self, features, mask, type="train"):
        
        current_device = mask.device
        
        # Choose a random class for each batch
        classes = torch.unique(mask)
        con_loss = torch.tensor([0.], device=current_device)
        
        index_masks = [einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=self.patch_size*(2**i), pw=self.patch_size*(2**i)).to(torch.float) for i in self.contrastive_indices]
        
        for c in classes:
        
            for i in self.contrastive_indices.to(current_device):
                
                index_mask = index_masks[i]
                
                if c not in index_mask:
                    break
                
                index_mask = torch.where(index_mask == c, 1., 0.).mean(1)
                feats = features[i]
                
                if self.reduced_contrastive_patch > 1:
                    patch_size = feats.size(2) // self.reduced_contrastive_patch
                    feats, _, index_mask = random_patch(feats, feats, index_mask.unsqueeze(1), patch_size)
                
                feats = einops.rearrange(feats, "b c h w -> b h w c")
                
                index_mask = index_mask.squeeze(1)
                    
                pos_feats = feats[index_mask == 1]
                neg_feats = feats[index_mask == 0]
                
                if self.loss_type == "contrastive":
                    con_loss += self.binary_contrastive_loss(pos_feats, neg_feats, current_device)
                elif self.loss_type == "info_nce":
                    con_loss += self.binary_info_nce_loss(pos_feats, neg_feats, current_device)
            
            con_loss = con_loss / len(self.contrastive_indices)
            
            return con_loss        
    
    def alt_binary_contrastive(self, features, mask, type="train"):
        
        current_device = mask.device
        
        # Choose a random class for each batch
        classes = torch.unique(mask)
        con_loss = torch.tensor([0.], device=current_device)
        
        index_masks = [einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=self.patch_size*(2**i), pw=self.patch_size*(2**i)).to(torch.float) for i in self.contrastive_indices]
        
        for i in self.contrastive_indices.to(current_device):
            
            index_mask = index_masks[i]
            feats = features[i]
            
            index_mask = einops.rearrange(index_mask, "b c h w -> (b h w) c")
            feats = einops.rearrange(feats, "b c h w -> (b h w) c")

            feats = self.projection[i](feats)
            
            num_channels = index_mask.size(-1)
            
            for c in classes:
                
                cindex_mask = (index_mask + 1) / (c + 1)
                
                cindex_mask = cindex_mask.sum(-1)
                
                #chosen_masks = index_mask[cindex_mask == num_channels]
                class_features = feats[cindex_mask == num_channels]
                
                if class_features.size(0) < 5:
                    continue
                  
                not_class_features = feats[cindex_mask != num_channels] 
                
                if self.loss_type == "contrastive":
                    con_loss = con_loss + self.binary_contrastive_loss(class_features, not_class_features, current_device)
                elif self.loss_type == "info_nce":
                    con_loss = con_loss + self.binary_info_nce_loss(class_features, not_class_features, current_device)
                elif self.loss_type == "supcon":
                    con_loss += self.binary_supcon_loss(class_features, not_class_features, current_device)    
                    
        con_loss = con_loss / len(self.contrastive_indices) / len(classes)
            
        return con_loss
    
    def alt_contrastive(self, features, mask, type="train"):
        
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
                    con_loss = con_loss + self.binary_contrastive_loss(class_features, not_class_features, current_device)
                elif self.loss_type == "info_nce":
                    con_loss = con_loss + self.binary_info_nce_loss(class_features, not_class_features, current_device)
                elif self.loss_type == "supcon":
                    con_loss += self.binary_supcon_loss(class_features, not_class_features, current_device)       
                    
        con_loss = con_loss / len(self.contrastive_indices) / len(classes)
            
        return con_loss
    
        
    def contrastive(self, features, mask, type="train"):
        
        current_device = mask.device
        
        # Choose a random class for each batch
        classes = torch.unique(mask)
        classes_mask = classes != 6
        classes = torch.masked_select(classes, classes_mask)
        con_loss = torch.tensor([0.], device=current_device)
        
        index_masks = [einops.rearrange(mask, "b (h ph) (w pw) -> b (ph pw) h w", ph=self.patch_size*(2**i), pw=self.patch_size*(2**i)).to(torch.float) for i in self.contrastive_indices]
        
        for c in classes:
        
            for i in self.contrastive_indices.to(current_device):
                
                index_mask = index_masks[i]
                
                if c not in index_mask:
                    break
                
                index_mask = torch.where(index_mask == c, 1., 0.).mean(1)
                feats = features[i]
                
                if self.reduced_contrastive_patch > 1:
                    patch_size = feats.size(2) // self.reduced_contrastive_patch
                    feats, _, index_mask = random_patch(feats, feats, index_mask.unsqueeze(1), patch_size)
                
                feats = einops.rearrange(feats, "b c h w -> b h w c")
                
                index_mask = index_mask.squeeze(1)
                    
                pos_feats = feats[index_mask == 1]
                neg_feats = feats[index_mask == 0]
                
                if self.loss_type == "contrastive":
                    con_loss += self.binary_contrastive_loss(pos_feats, neg_feats, current_device)
                elif self.loss_type == "info_nce":
                    con_loss += self.binary_info_nce_loss(pos_feats, neg_feats, current_device)
                    
        con_loss = con_loss / len(self.contrastive_indices)
            
        return con_loss
            
        """
            ncmask, cmask, tempmask = self.cas[i](mask, c, self.pos_param, self.neg_param)
            feats = features[i]
            
            cfeats = feats + cmask
            ncfeats = feats + ncmask
            
            # Upscale so that the values from the last iteration can be added
            #if i > self.contrastive_indices[0]:
            #    last_class_attention = self.upscale[i - 1](last_class_attention)
            #    last_not_class_attention = self.upscale[i - 1](last_not_class_attention)
            #    cfeats = cfeats + last_class_attention
            #    ncfeats = ncfeats + last_not_class_attention
            
            cfeats = einops.rearrange(cfeats, "b c h w -> b h w c")
            ncfeats = einops.rearrange(ncfeats, "b c h w -> b h w c")
            
            class_attention = self.layer_norm[i](cfeats)
            not_class_attention = self.layer_norm[i](ncfeats)
            
            class_attention = einops.rearrange(class_attention, "b h w c -> b c h w")
            not_class_attention = einops.rearrange(not_class_attention, "b h w c -> b c h w")
            
            # Get a random patch of patches
            patch_size = class_attention.size(2) // 2
            class_attention, not_class_attention, mask_patch = random_patch(class_attention, not_class_attention, tempmask, patch_size)
            
            class_attention, pos_offset, pos_pos, _ = self.dattention[i](class_attention)
            not_class_attention, neg_offset, neg_pos, _ = self.dattention[i](not_class_attention)
            
            #cattn = einops.rearrange(cattn, 'b c h w -> b h w c')
            #ncattn = einops.rearrange(ncattn, 'b c h w -> b h w c')
            
            #attn_loss = self.attention(cattn, ncattn, cfeats, ncfeats, idx=i, type=type)
            
            #loss = loss + attn_loss
            
            #last_class_attention = class_attention
            #last_not_class_attention = not_class_attention
            
            class_attention = einops.rearrange(class_attention, "b c h w -> b h w c")
            not_class_attention = einops.rearrange(not_class_attention, "b c h w -> b h w c")
            
            # L2 Norm
            class_attention = F.normalize(class_attention, p=2, dim=-1)
            not_class_attention = F.normalize(not_class_attention, p=2, dim=-1)
            
            # flatten before projection
            class_attention = class_attention.flatten(0, -2).contiguous()
            not_class_attention = not_class_attention.flatten(0, -2).contiguous()

            class_attention = self.projection[i](class_attention)
            not_class_attention = self.projection[i](not_class_attention)

            num_patches = class_attention.size(0)
            
            indices = torch.arange(num_patches, device=current_device)
            
            # split the indices equally between them
            #pos_nc_feat_indices, neg_nc_feat_indices = split_tensor(indices) 
            #pos_c_feat_indices, neg_c_feat_indices = split_tensor(indices)
            #
            #neg_nclass_features = not_class_attention[neg_nc_feat_indices]
            #pos_nclass_features = not_class_attention[pos_nc_feat_indices]
            #
            #neg_class_features = class_attention[neg_c_feat_indices]
            #pos_class_features = class_attention[pos_c_feat_indices]
            
            topk_start_percentage = 0.0
            topk_end_percentage = topk_start_percentage + 0.4
            count_min = 0
            
            con_loss = torch.tensor([0.], device=current_device)
            
            similarity_dim = -1
            
            # Negative similarity
            similarities = F.cosine_similarity(class_attention, not_class_attention, dim=similarity_dim)
            similarities, _  = torch.sort(similarities, descending=True)
            similarities = similarities[int(similarities.size(0) * topk_start_percentage):int(similarities.size(0) * topk_end_percentage)]
            distances = 1 - similarities
            con_loss += self.cl(distances, label=0)
            
            # Positive similarity
            
            indices = torch.arange(num_patches, device=current_device)
            
            first_class_indices, second_class_indices = split_tensor(indices)
            first_nclass_indices, second_nclass_indices = split_tensor(indices)
            
            similarities = F.cosine_similarity(class_attention[first_class_indices], class_attention[second_class_indices], dim=similarity_dim)
            similarities, _  = torch.sort(similarities, descending=False)
            similarities = similarities[int(similarities.size(0) * topk_start_percentage):int(similarities.size(0) * topk_end_percentage)]
            distances = 1 - similarities
            con_loss += self.cl(distances, label=1)
            
            similarities = F.cosine_similarity(not_class_attention[first_nclass_indices], not_class_attention[second_nclass_indices], dim=similarity_dim)
            similarities, _  = torch.sort(similarities, descending=False)
            similarities = similarities[int(similarities.size(0) * topk_start_percentage):int(similarities.size(0) * topk_end_percentage)]
            distances = 1 - similarities
            con_loss += self.cl(distances, label=1)

            loss = loss + con_loss
         
        return loss
        """


if __name__ == "__main__":
    
    config = py2cfg("config/potsdam/contrastive_dcswin.py")
    
    model = ContrastiveDCSwin(config)
    