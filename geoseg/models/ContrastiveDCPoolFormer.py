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
from .poolformer import PoolFormer

from scipy.ndimage import distance_transform_edt as distance
from torchvision import transforms
from functools import partial
from operator import itemgetter

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU()
        )

def extract_edges_batch(masks, class_id):
    assert masks.dim() == 3, "masks should have 3 dimensions: N x H x W"

    if torch.cuda.is_available():
        masks = masks.cuda()

    # Create a binary mask for the specified class
    binary_masks = (masks == class_id).float()

    # Calculate the difference between neighboring pixels
    horizontal_diff = binary_masks[:, :, 1:] - binary_masks[:, :, :-1]
    vertical_diff = binary_masks[:, 1:, :] - binary_masks[:, :-1, :]

    # Create the edge masks
    edge_masks = torch.zeros_like(binary_masks)
    edge_masks[:, :, 1:] += horizontal_diff.abs()
    edge_masks[:, :, :-1] += horizontal_diff.abs()
    edge_masks[:, 1:, :] += vertical_diff.abs()
    edge_masks[:, :-1, :] += vertical_diff.abs()

    # Convert the edge masks to boolean tensors
    edge_masks = (edge_masks > 0)
    
    edge_masks = edge_masks.to(torch.int32)

    return edge_masks

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

class ContrastiveDCPoolFormer(nn.Module):
    def __init__(self, config, attn_loss, contrastive_loss) -> None:
        super().__init__()
        
        self.config = config
        self.contrastive_indices = torch.tensor(config["con_conf"]["contrastive_indices"])
        self.con_loss_clip = config["con_conf"]["con_loss_clip"]
        self.reduced_contrastive_patch = config["con_conf"]["reduced_contrastive_patch"]
        self.take_top = config["con_conf"]["take_top"]
        self.max_samples = config["con_conf"]["max_samples"]
        self.num_samples = config["con_conf"]["num_samples"]
        
        self.ignore_index = config["con_conf"]["ignore_index"]
        
        self.patch_size = config["pool_conf"]["in_stride"]
        
        self.backbone = PoolFormer(**config["pool_conf"])
        
        self.loss_type = config["con_conf"]["loss_type"]
        
        load_checkpoint(self.backbone, "pretrain_weights/poolformer_s12.pth.tar")
        
        self.decoder = Decoder(**config["decoder_conf"])
        
        self.projection = nn.ModuleList(
                [ProjectionLayer(
                    in_dim, 
                    config["proj_conf"]["out_features"]
                ) for in_dim in config["proj_conf"]["in_features"]
            ])
        
        self.cl = contrastive_loss
        
        self.con_up = nn.UpsamplingBilinear2d(scale_factor=self.patch_size)
        self.pixel_projection = ProjectionLayer(config["decoder_conf"]["encoder_channels"][0], config["proj_conf"]["out_features"])
        self.convbn = ConvBNReLU(config["decoder_conf"]["encoder_channels"][0], config["decoder_conf"]["encoder_channels"][0])
        
    def forward(self, x, mask=None, type="train"):
        features = self.backbone(x)
        
        x, con_x = self.decoder(*features)

        if mask == None:
            return x
        
        contrastive_loss = self.contrastive(features=features, mask=mask, predictions=x, con_x=con_x)

        return contrastive_loss, x
    
    def sample(self, low_pos, high_pos, neg):
        
        num_low = low_pos.size(0)
        num_high = high_pos.size(0)
        num_neg = neg.size(0)
        
        return None

    def contrastive_loss(self, low_pos, high_pos, neg, current_device):
        
        con_loss = torch.tensor([0.], device=current_device)
        
        num_low = low_pos.size(0)
        num_high = high_pos.size(0)
        num_neg = neg.size(0)
        
        low_indices = torch.arange(num_low, device=current_device)
        high_indices = torch.arange(num_high, device=current_device)
        neg_indices = torch.arange(num_neg, device=current_device)
        
        low_high_pairs = torch.cartesian_prod(low_indices, high_indices)
        high_neg_pairs = torch.cartesian_prod(high_indices, neg_indices)
        
        low_high_similarity = torch.cosine_similarity(low_pos[low_high_pairs[:, 0]], high_pos[low_high_pairs[:, 1]])
        high_neg_similarity = torch.cosine_similarity(high_pos[high_neg_pairs[:, 0]], neg[high_neg_pairs[:, 1]])

        _, low_high_indices = torch.sort(low_high_similarity, descending=False)
        _, high_neg_indices = torch.sort(high_neg_similarity, descending=True)
        
        num_examples = self.num_samples * 4
        
        pos_low_pos = low_pos[low_high_pairs[low_high_indices[:num_examples]][:, 0]]
        pos_high_pos = high_pos[low_high_pairs[low_high_indices[:num_examples]][:, 1]]
    
        if num_low >= 2 and num_high >= 2:
            
            con_loss = con_loss + self.cl(pos_high_pos, pos_low_pos, 1)
        
        
        neg_high = high_pos[high_neg_pairs[high_neg_indices[:num_examples]][:, 0]]
        neg_neg = neg[high_neg_pairs[high_neg_indices[:num_examples]][:, 1]]
        
        if num_low >= 2 and num_neg >= 2:
            
            con_loss = con_loss + self.cl(neg_high, neg_neg, 0)
        
            
        return con_loss
        
    def info_nce_loss(self, low_pos, high_pos, neg, current_device):
        
        con_loss = torch.tensor([0.], device=current_device)
        
        num_low = low_pos.size(0)
        num_high = high_pos.size(0)
        num_neg = neg.size(0)
        
        low_indices = torch.arange(num_low, device=current_device)
        high_indices = torch.arange(num_high, device=current_device)
        neg_indices = torch.arange(num_neg, device=current_device)
        
        high_neg_pairs = torch.cartesian_prod(high_indices, neg_indices)
        
        high_neg_similarity = torch.cosine_similarity(high_pos[high_neg_pairs[:, 0]], neg[high_neg_pairs[:, 1]])

        _, high_neg_indices = torch.sort(high_neg_similarity, descending=True)
        
        num_examples = self.num_samples * 4
        
        high_pos_indices = torch.unique(high_neg_pairs[high_neg_indices[:num_examples]][:, 0])
        neg_indices = torch.unique(high_neg_pairs[high_neg_indices[:num_examples]][:, 1]) 
        
        low_pos = low_pos[:high_pos_indices.size(0)]
        high_pos = high_pos[high_pos_indices]
        neg = neg[neg_indices]

        con_loss = con_loss + self.cl(high_pos, low_pos, neg)

        return con_loss

    
    def contrastive(self, features, mask, predictions, con_x):
        
        predictions = F.softmax(predictions, dim=1)
        
        con_x = self.convbn(con_x)
        con_x = self.con_up(con_x)

        current_device = mask.device
        
        # Choose a random class for each batch
        classes = torch.unique(mask)
        classes_mask = classes != self.ignore_index
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
            
            for c in classes:
                # B, C, PH, PW
                
                predc = c
                
                temp_cpreds = preds[:, predc]
                # B, 1, H, W
                temp_cpreds = einops.rearrange(temp_cpreds, "b c h w -> (b h w) c")
                
                temp_cpreds = temp_cpreds.mean(-1)
                
                cindex_mask = torch.where(index_mask == c, 1, 0)
                cindex_mask = cindex_mask.to(torch.float32).mean(-1)
                
                # We only want cpreds where class distribution is homogenous
                cpreds = temp_cpreds[cindex_mask == 1]
                class_features = feats[cindex_mask == 1]                
                
                _, cpred_indices = torch.sort(cpreds, descending=False)
                
                index = int(len(cpred_indices) * self.take_top)
                
                index = min(index, self.num_samples)

                low_cpred_indices = cpred_indices[:index]
                high_cpred_indices = cpred_indices[-index:]
                
                low_class_features = class_features[low_cpred_indices]
                high_class_features = class_features[high_cpred_indices]
                
                num_low, num_high = low_class_features.size(0), high_class_features.size(0)
                
                if num_low < 10 or num_high < 10:
                    continue
                
                not_class_features = feats[cindex_mask == 0.]
                # Choose the patches where the prediction-percentages are highest
                not_cpreds = temp_cpreds[cindex_mask == 0.]
                
                _, not_cpreds_indices = torch.sort(not_cpreds, descending=True)

                index = int(len(cpred_indices) * self.take_top)
                index = min(index, self.num_samples)
                
                not_class_features = not_class_features[not_cpreds_indices[:index]]

                num_neg = not_class_features.size(0)
                                
                if num_neg < 10:
                    continue
                
                if self.loss_type == "contrastive":
                    con_loss = con_loss + self.contrastive_loss(low_class_features, high_class_features, not_class_features, current_device)
                elif self.loss_type == "info_nce":
                    con_loss = con_loss + self.info_nce_loss(low_class_features, high_class_features, not_class_features, current_device)
                
        """
        # This does not require all stages, as if we've covered the smallest patches, then everything should be covered
        con_x = einops.rearrange(con_x, "b c (h ph) (w pw) -> b h w ph pw c", ph=self.patch_size*(2**0), pw=self.patch_size*(2**0))
        preds = predictions[0]
        index_mask = einops.rearrange(index_masks[0], "b (ph pw) h w -> b h w ph pw",  ph=self.patch_size*(2**0), pw=self.patch_size*(2**0))
        
        for c in classes:
            
            # Predictions for target class to collect high and low confidence predictions
            cpreds = preds[:, c]
            cpreds = einops.rearrange(cpreds, "b (ph pw) h w -> b h w ph pw",  ph=self.patch_size*(2**0), pw=self.patch_size*(2**0))
            
            cindex_mask = torch.where(index_mask == c, 1, 0).to(torch.float32)
            avg_cindex_mask = cindex_mask.mean((-2, -1))
            
            patch_mask = (avg_cindex_mask != 0) & (avg_cindex_mask != 1)
            
            patch_pixel_confidence = cpreds[patch_mask]
            patch_pixel_representations = con_x[patch_mask]
            positive_patch_masks = cindex_mask[patch_mask]
            negative_patch_masks = torch.logical_not(positive_patch_masks)
            
            
            patch_mask_edges = extract_edges_batch(positive_patch_masks, 1)
            
            positive_pixel_mask = torch.logical_and(patch_mask_edges, positive_patch_masks)
            negative_pixel_mask = torch.logical_and(patch_mask_edges, negative_patch_masks)
            
            patch_pixel_representations = self.pixel_projection(patch_pixel_representations)
                        
            positive_edge_pixel_confidence = patch_pixel_confidence[positive_pixel_mask]
            positive_edge_pixels = patch_pixel_representations[positive_pixel_mask]
            negative_edge_pixels = patch_pixel_representations[negative_pixel_mask]
            
            _, confidence_indices = torch.sort(positive_edge_pixel_confidence)
            index = int(len(confidence_indices) * self.take_top)
            
            low_cpred_indices = confidence_indices[:index // 2]
            high_cpred_indices = confidence_indices[-index * 2:]
            
            low_pixel_features = positive_edge_pixels[low_cpred_indices]
            high_pixel_features = positive_edge_pixels[high_cpred_indices]
            
            if self.loss_type == "contrastive":
                temp_loss = self.contrastive_loss(low_pixel_features, high_pixel_features, negative_edge_pixels, current_device)
            elif self.loss_type == "info_nce":
                temp_loss = self.info_nce_loss(low_pixel_features, high_pixel_features, negative_edge_pixels, current_device)
                
            con_loss = con_loss + temp_loss
        
        """
        con_loss = con_loss / len(self.contrastive_indices)
        con_loss = con_loss / len(classes)
            
        return con_loss


if __name__ == "__main__":
    
    config = py2cfg("config/potsdam/contrastive_dcswin.py")
    
    model = ContrastiveDCSwin(config)
    