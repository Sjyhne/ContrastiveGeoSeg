from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.ContrastiveUnetFormer import ContrastiveUnetFormer

from config.common.dataset_specific import train_aug, val_aug

from info_nce import InfoNCE

check_val_every_n_epoch = 1
gpus = [0, 1]
accumulate_grad_batches = 1

train_batch_size = 4 * len(gpus)
val_batch_size = 1

auto_lr_find = True

# Contrastive conf
con_loss_clip = 1.0
gradient_clip_val = 1.0

contrastive = None # Use None for False
contrastive_indices = [0, 1, 2, 3]
loss_type = "contrastive"
reduced_contrastive_patch = 1
soft_ce_smoothing = 0.1
take_top = 0.5

# weights and biases
wandb_logging_name = f"UnetFormer Contrastive (bs {train_batch_size})"
wandb_project = f"CT-UnetFormer"
wandblogger = True

weights_name = wandb_logging_name
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = weights_name
log_name = 'potsdam_full/{}'.format(weights_name)

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
window_size = 7
mlp_ratio = 4
qkv_bias = True
swin_drop_rate = 0
swin_attn_drop_rate = 0

swin_conf = {
    "pretrain_img_size": image_size,
    "patch_size": patch_size,
    "in_chans": in_channels,
    "embed_dim": embedding_dims[0],
    "depths": swin_depths,
    "num_heads": swin_num_heads,
    "window_size": window_size,
    "mlp_ratio": mlp_ratio,
    "qkv_bias": qkv_bias,
    "drop_rate": swin_drop_rate,
    "attn_drop_rate": swin_attn_drop_rate,
}

# Decoder config
dropout = 0.05
atrous_rates = (6, 12)


decoder_conf = {
    "encoder_channels": [i for i in embedding_dims],
    "decode_channels": 64,
    "num_classes": num_classes,    
    "dropout": dropout,
}

# Projection conf

proj_out = 128

proj_conf = {
    "in_features": [i for i in embedding_dims],
    "out_features": proj_out 
}

con_conf = {
    "contrastive_indices": contrastive_indices,
    "contrastive": contrastive,
    "con_loss_clip": con_loss_clip,
    "loss_type": loss_type,
    "reduced_contrastive_patch": reduced_contrastive_patch,
    "soft_ce_smoothing": soft_ce_smoothing,
    "take_top": take_top
}


config = {
    "swin_conf": swin_conf,
    "decoder_conf": decoder_conf,
    "proj_conf": proj_conf,
    "con_conf": con_conf,
    "learning_rate": learning_rate,
    "effective_bs": train_batch_size,
    "auto_lr_find": auto_lr_find
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
    
model = ContrastiveUnetFormer(config=config, attn_loss=attn_loss, contrastive_loss=con_loss)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0
)

"""
RUNTIME SPECIFIC
"""

max_epoch = 50
classes = CLASSES

monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 2
save_last = False

check_val_every_n_epoch = 1
gpus = [0]
accumulate_grad_batches = 1
train_batch_size = 16 * len(gpus)

strategy = "dp"

pretrained_ckpt_path = None
resume_ckpt_path = None

dp = 1.0

# Profiler
profiler = None

train_dataset = PotsdamDataset(
    data_root='data/potsdam/train', 
    mode='train',
    mosaic_ratio=0.25,
    transform=train_aug,
    data_percentage=dp
)

val_dataset = PotsdamDataset(
    data_root="data/potsdam/test", 
    transform=val_aug, 
    data_percentage=dp
)

test_dataset = PotsdamDataset(
    data_root="data/potsdam/test", 
    transform=val_aug, 
    data_percentage=dp
)
n_workers = 8

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