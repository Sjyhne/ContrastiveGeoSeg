from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.ade20k_dataset import *
from geoseg.models.ContrastiveDCPoolFormer import ContrastiveDCPoolFormer

from config.common.dataset_specific import ade20k_train_aug, ade20k_val_aug
from lightning.pytorch.strategies import SingleDeviceStrategy
from info_nce import InfoNCE

check_val_every_n_epoch = 1
gpus = [11]
accumulate_grad_batches = 1
learning_rate = 1e-4

train_batch_size = 8
val_batch_size = 1

auto_lr_find = True

# Contrastive conf
con_loss_clip = 1000
gradient_clip_val = None

contrastive = True # Use None for False
contrastive_indices = [0, 1, 2, 3]
loss_type = "contrastive"
reduced_contrastive_patch = 1
soft_ce_smoothing = 0.
take_top = 0.4
max_samples = 10_000
num_samples = 512
# weights and biases
wandb_logging_name = f"DCPoolFormer Contrastive {contrastive_indices} (num_samples {num_samples}) (clip {con_loss_clip}) (lr {learning_rate})"
wandb_project = f"CT-DCPoolFormer ADE20K (0.1)"
wandblogger = True

weights_name = wandb_logging_name
weights_path = "model_weights/ade20k/{}".format(weights_name)
test_weights_name = weights_name
log_name = 'ade20k_full/{}'.format(weights_name)


"""
MODEL SPECIFIC
"""

# Common config
embedding_dims = [128, 256, 512, 1024]
num_classes = CLASSES
image_size = 512
patch_size = 4

# Swin config
in_channels = 3
swin_depths = [2, 2, 18, 2]
swin_num_heads = [4, 8, 16, 32]
window_size = 8
mlp_ratio = 4
qkv_bias = True
swin_drop_rate = 0
swin_attn_drop_rate = 0


pool_conf = {
    "layers": swin_depths,
    "embed_dims": embedding_dims,
    "in_patch_size": patch_size * 2,
    "in_stride": patch_size,
    "in_pad": patch_size // 2,
    "num_classes": num_classes, 
    "mlp_ratios": [mlp_ratio for i in range(4)], 
    "downsamples": [True for i in range(4)],
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


# Projection conf

ignore_index = CLASSES
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
    "take_top": take_top,
    "max_samples": max_samples,
    "num_samples": num_samples,
    "ignore_index": ignore_index
}


config = {
    "pool_conf": pool_conf,
    "decoder_conf": decoder_conf,
    "proj_conf": proj_conf,
    "con_conf": con_conf,
    "learning_rate": learning_rate,
    "effective_bs": train_batch_size,
    "auto_lr_find": auto_lr_find
}



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss = nn.NLLLoss()

    def forward(self, output1, output2, label):
        similarity = torch.cosine_similarity(output1, output2, dim=-1).unsqueeze(-1)
        similarity = similarity + 1
        similarity = similarity / 2
        similarity = similarity + 1e-6
        
        #+ 1 / 2 # To get them in the range 0, 1
        
        #if label == 1:
        #    similarity, _ = torch.sort(similarity, descending=False)
        #else:
        #    similarity, _ = torch.sort(similarity, descending=True)
        
        neg_sim = 1 - similarity
        
        similarity = torch.cat([neg_sim, similarity], dim=-1)
        
        if label == 1:
            labels = torch.ones(similarity.size(0), device=output1.device, dtype=torch.long)
        else:
            labels = torch.zeros(similarity.size(0), device=output1.device, dtype=torch.long)
        
        similarity = torch.log(similarity)
        
        loss = self.loss(similarity, labels)
        
        if torch.isnan(loss):
            loss = torch.tensor([0.], device=output1.device)

        return loss

if loss_type == "contrastive":
    con_loss = ContrastiveLoss()
elif loss_type == "info_nce":
    con_loss = InfoNCE()
elif loss_type == "supcon":
    con_loss = SupConLoss()
    
model = ContrastiveDCPoolFormer(config=config, attn_loss=None, contrastive_loss=con_loss)

loss = JointLoss(
    SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
    DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0
)

"""
RUNTIME SPECIFIC
"""

max_epoch = 40
classes = CLASSES

monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 2
save_last = False

strategy = "ddp_find_unused_parameters_true"

pretrained_ckpt_path = None
resume_ckpt_path = None

dp = 0.1

# Profiler
profiler = None

train_dataset = ADE20KDataset(
    data_root='data/ade20k/train', 
    mode='train',
    transform=ade20k_train_aug,
    data_percentage=dp
)

val_dataset = ADE20KDataset(
    data_root="data/ade20k/test",
    transform=ade20k_val_aug,
    data_percentage=dp
)

test_dataset = ADE20KDataset(
    data_root="data/ade20k/test", 
    transform=ade20k_val_aug,
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