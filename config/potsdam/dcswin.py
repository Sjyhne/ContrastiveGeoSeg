from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.potsdam_dataset import *
from geoseg.models.DCSwin import dcswin_small, dcswin_base
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from timm.scheduler.poly_lr import PolyLRScheduler
from lightning.pytorch.accelerators import find_usable_cuda_devices

# training hparam
max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 1
val_batch_size = 1
lr = 1e-3
weight_decay = 2.5e-4
backbone_lr = 1e-4
backbone_weight_decay = 2.5e-4
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES

gradient_clip_val = 1.0
contrastive = False
wandb_logging_name = f"Contrastive DCSwin"
wandb_project = "Contrastive DCSwin Full"
wandblogger = True

contrastive = None

weights_name = "dcswin-small-1024-ms-512crop-e30"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "dcswin-small-1024-ms-512crop-e30"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1
gpus = [8, 9, 10, 11, 12, 13, 14, 15] #find_usable_cuda_devices(8)
strategy = "ddp"
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
model = dcswin_base(num_classes=num_classes)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
# loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


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

def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


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


profiler = None

train_dataset = PotsdamDataset(data_root='data/potsdam/train', mode='train',
                               mosaic_ratio=0.25, transform=train_aug, data_percentage=1.0)

val_dataset = PotsdamDataset(transform=val_aug, data_percentage=1.0)
test_dataset = PotsdamDataset(data_root='data/potsdam/test',
                              transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=16,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=16,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(model, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
