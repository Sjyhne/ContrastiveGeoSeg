from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.ContrastiveDCCSwin import ContrastiveDCCSwin
from catalyst.contrib.nn import Lookahead

from timm.models.layers import to_2tuple

from info_nce import InfoNCE

"""
MODEL SPECIFIC
"""

# Common config
embedding_dims = [64, 128, 256, 512]
num_classes = len(CLASSES)
image_size = 512
patch_size = 4
learning_rate = 8e-5


# Swin config
in_channels = 3
swin_depths = [2, 2, 8, 2]
swin_num_heads = [2, 4, 8, 16]
window_size = 8
mlp_ratio = 4
qkv_bias = True
swin_drop_rate = 0
swin_attn_drop_rate = 0


cswin_conf = {
    "img_size": image_size,
    "in_chans": 3,
    "num_classes": num_classes,
    "head_dim": embedding_dims[0],
    "embed_dim": embedding_dims,
    "layers": [2, 2, 8, 2],
}

# Decoder config
dropout = 0.05
atrous_rates = (6, 12)


decoder_conf = {
    "encoder_channels": [i for i in embedding_dims],
    "num_classes": num_classes,    
    "dropout": dropout,
    "atrous_rates": atrous_rates,
    "patch_size": patch_size}

# Class Attn config
attn_depths = [1, 1, 1, 1]
attn_in_channels = 1
attn_num_heads = swin_num_heads
attn_drop_rate = swin_drop_rate
attn_attn_drop_rate = swin_attn_drop_rate
attn_window_size = window_size
attn_mlp_ratio = mlp_ratio

class_attn_conf = {
    "embed_dims": embedding_dims,
    "swin_conf": {
        "pretrain_img_size": image_size,
        "patch_size": patch_size,
        "in_chans": attn_in_channels,
        "embed_dim": embedding_dims[0],
        "depths": attn_depths,
        "num_heads": swin_num_heads,
        "window_size": window_size,
        "mlp_ratio": mlp_ratio,
        "qkv_bias": qkv_bias,
        "drop_rate": swin_drop_rate,
        "attn_drop_rate": swin_attn_drop_rate,
    }
}

# Projection conf

proj_out = 128

proj_conf = {
    "in_features": [i for i in embedding_dims],
    "out_features": proj_out 
}

# LayerNorm conf

norm_conf = {
    "embedding_dims": [i for i in embedding_dims]
}

# Deformable Attention
dattn_heads = [2, 4, 8, 16]
n_head_channels = [(embedding_dims[i]) // dattn_heads[i] for i in range(len(embedding_dims))]
groups = [1, 2, 4, 8]
dattn_drop = 0.
dattn_proj_drop = 0.
strides = [1, 1, 1, 1]
offset_range_factor = [2, 2, 2, 2]
use_pe = False
dwc_pe = False
no_off = False
fixed_pe = False

dattn_conf = {
        "feature_map_size": to_2tuple(image_size),
        "dimensions": embedding_dims,
        "heads": dattn_heads,
        "n_head_channels": n_head_channels,
        "groups": groups,
        "attn_drop": dattn_drop, # Fails when not 0.0....
        "proj_drop": dattn_proj_drop,
        "strides": strides,
        "offset_range_factor": offset_range_factor,
        "use_pe": use_pe,
        "dwc_pe": dwc_pe,
        "no_off": no_off,
        "fixed_pe": fixed_pe
    }

# Contrastive conf

gradient_clip_val = 1.0
contrastive = True # Use None for False
dist_loss = True
contrastive_indices = [0, 1, 2, 3]
con_loss_clip = 1.0
attn_loss_clip = None
binary = None # Use None for False
loss_type = "info_nce"
reduced_contrastive_patch = 1
soft_ce_smoothing = 0.1
take_top = 0.5

con_conf = {
    "contrastive_indices": contrastive_indices,
    "dist_loss": dist_loss,
    "contrastive": contrastive,
    "con_loss_clip": con_loss_clip,
    "attn_loss_clip": attn_loss_clip,
    "binary": binary,
    "loss_type": loss_type,
    "reduced_contrastive_patch": reduced_contrastive_patch,
    "soft_ce_smoothing": soft_ce_smoothing,
    "take_top": take_top
}


config = {
    "cswin_conf": cswin_conf,
    "decoder_conf": decoder_conf,
    "class_attn_conf": class_attn_conf,
    "proj_conf": proj_conf,
    "norm_conf": norm_conf,
    "dattn_conf": dattn_conf,
    "con_conf": con_conf
}

ignore_index = len(CLASSES)

attn_loss = torch.nn.HuberLoss()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.softce = SoftCrossEntropyLoss(smooth_factor=soft_ce_smoothing)

    def forward(self, output1, output2, label):
        similarity = (torch.cosine_similarity(output1, output2, dim=-1).unsqueeze(-1) + 1) / 2 # To get them in the range 0, 1
        
        if label == 1:
            similarity, _ = torch.sort(similarity, descending=False)
        else:
            similarity, _ = torch.sort(similarity, descending=True)
            
        similarity = similarity[:int(len(similarity) * take_top)]
        
        sim_dupe = 1 - similarity.clone()
        similarity = torch.cat([sim_dupe, similarity], dim=-1)
        
        if label == 1:
            labels = torch.ones(similarity.size(0), device=output1.device, dtype=torch.long)
        else:
            labels = torch.zeros(similarity.size(0), device=output1.device, dtype=torch.long)

        seg_logits = self.softce(similarity, labels)

        return seg_logits

if loss_type == "contrastive":
    con_loss = ContrastiveLoss()
elif loss_type == "info_nce":
    con_loss = InfoNCE()
elif loss_type == "supcon":
    con_loss = SupConLoss()
    
model = ContrastiveDCCSwin(config=config, attn_loss=attn_loss, contrastive_loss=con_loss)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0
)

"""
RUNTIME SPECIFIC
"""

max_epoch = 50
val_batch_size = 1
classes = CLASSES

monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 2
save_last = False

check_val_every_n_epoch = 1
gpus = [12, 13]
accumulate_grad_batches = 1
train_batch_size = 16 * len(gpus)

strategy = "dp"

pretrained_ckpt_path = None
resume_ckpt_path = None


dp = 1.0

# weights and biases
wandb_logging_name = f"DCUniFormer InfoNCE w/sampling & clipping NEW"
wandb_project = f"Potsdam DCUniFormer {dp}"
wandblogger = True

weights_name = wandb_logging_name
weights_path = "model_weights/potsdam_full/{}".format(weights_name)
test_weights_name = weights_name
log_name = 'potsdam_full/{}'.format(weights_name)

# Profiler
profiler = None

"""
MISC
"""
use_aux_loss = False


"""
DATASET SPECIFIC
"""

def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)

def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=375, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img = img.resize(ORIGIN_IMG_SIZE)
    mask = mask.resize(ORIGIN_IMG_SIZE)
    img, mask = np.array(img), np.array(mask)
    if np.isin(7, mask):
        print("Found 7 - fixing")
        mask[mask == 7] = 6
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

def val_aug(img, mask):
    img = img.resize(ORIGIN_IMG_SIZE)
    mask = mask.resize(ORIGIN_IMG_SIZE)
    img, mask = np.array(img), np.array(mask)
    if np.isin(7, mask):
        print("Found 7 - fixing")
        mask[mask == 7] = 6
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask

train_dataset = PotsdamDataset(
    data_root='data/potsdam/train', 
    mode='train',
    mosaic_ratio=0.25,
    transform=train_aug,
    data_percentage=dp
)

val_dataset = PotsdamDataset(data_root="data/potsdam/test", transform=val_aug, data_percentage=dp)

n_workers = 12

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=n_workers,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=n_workers // 2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

"""
OPTIMIZER
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)