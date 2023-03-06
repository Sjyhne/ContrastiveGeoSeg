import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from tools.cfg import py2cfg
import os
import torch
from torch import nn
import cv2
import numpy as np
import argparse
from pathlib import Path
from tools.metric import Evaluator
from pytorch_lightning.loggers import CSVLogger, WandbLogger
import random
import einops
import torch.nn.functional as F

from pytorch_lightning.utilities import rank_zero_only

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


def split_tensor(tensor):
    indices = torch.randperm(tensor.numel())
    half = indices.numel()//2
    return tensor.view(-1)[indices[:half]], tensor.view(-1)[indices[half:]]

class Supervision_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = config.model
        self.automatic_optimization = True

        self.loss_fn = config.loss
        
        if config.contrastive != None:
            self.contrastive = True
        else:
            self.contrastive = False
        
        self.metrics_train = Evaluator(num_class=config.num_classes)
        self.metrics_val = Evaluator(num_class=config.num_classes)

    def forward(self, x, mask=None):
        # only net is used in the prediction/inference
        
        if mask != None and self.contrastive:
            contrastive_loss, x = self.model(x, mask)
            return contrastive_loss, x
        else:
            x = self.model(x)
            return x

    def training_step(self, batch, batch_idx):
        img, mask = batch['img'], batch['gt_semantic_seg']
        
        #opt = self.optimizers()
        #opt = opt.optimizer
        #opt.zero_grad()
        
        loss = torch.tensor([0.], device=self.device)
        
        if self.contrastive:
            contrastive_loss, prediction = self.model(img, mask, "train")
            contrastive_loss = torch.clip(contrastive_loss, 0.0, self.config.con_loss_clip)
            self.log("train_contrastive_loss", contrastive_loss.item(), on_epoch=True, batch_size=self.config.train_batch_size, sync_dist=True)
            loss = loss + contrastive_loss
        else:
            prediction = self.model(img)
        
        loss = loss + self.loss_fn(prediction, mask)
        
        if self.config.use_aux_loss:
            pre_mask = nn.Softmax(dim=1)(prediction[0])
        else:
            pre_mask = nn.Softmax(dim=1)(prediction)

        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_train.add_batch(mask[i].detach().cpu().numpy(), pre_mask[i].detach().cpu().numpy())
        
        self.log("train_loss", loss.item(), on_epoch=True, batch_size=self.config.train_batch_size, sync_dist=True)
        
        #print("loss:", loss)
        # supervision stage
        #self.manual_backward(loss)
        #if (batch_idx + 1) % self.config.accumulate_n == 0:
        #    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_val)
        #opt.step()

        #sch = self.lr_schedulers()
        #if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
        #    sch.step()
        
        return loss
    
    def training_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_train.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_train.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_train.F1())

        OA = np.nanmean(self.metrics_train.OA())
        iou_per_class = self.metrics_train.Intersection_over_Union()
        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        print('\ntrain:', eval_value)

        iou_value = {}
        
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        
        print(iou_value)
        self.metrics_train.reset()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log_dict = {"train_loss": loss, 'train_mIoU': mIoU, 'train_F1': F1, 'train_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        img, mask = batch['img'].cuda(), batch['gt_semantic_seg'].cuda()
        
        loss_val = torch.tensor([0.], device=self.device)
        
        if self.contrastive:
            contrastive_loss, prediction = self.model(img, mask, "val")
            self.log("val_contrastive_loss", contrastive_loss.item(), on_epoch=True, batch_size=self.config.train_batch_size, sync_dist=True)
            loss_val = loss_val + contrastive_loss
        else:
            prediction = self.model(img)
        
        loss_val = loss_val + self.loss_fn(prediction, mask)
        
        pre_mask = nn.Softmax(dim=1)(prediction)
        pre_mask = pre_mask.argmax(dim=1)
        for i in range(mask.shape[0]):
            self.metrics_val.add_batch(mask[i].detach().cpu().numpy(), pre_mask[i].detach().cpu().numpy())
        
        self.log("val_loss", loss_val.item(), on_epoch=True, batch_size=self.config.train_batch_size, sync_dist=True)
        
        return loss_val

    def validation_epoch_end(self, outputs):
        if 'vaihingen' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'potsdam' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'whubuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'massbuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        elif 'inriabuilding' in self.config.log_name:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union()[:-1])
            F1 = np.nanmean(self.metrics_val.F1()[:-1])
        else:
            mIoU = np.nanmean(self.metrics_val.Intersection_over_Union())
            F1 = np.nanmean(self.metrics_val.F1())

        OA = np.nanmean(self.metrics_val.OA())
        iou_per_class = self.metrics_val.Intersection_over_Union()

        eval_value = {'mIoU': mIoU,
                      'F1': F1,
                      'OA': OA}
        
        print('\nval:', eval_value)
        iou_value = {}
        for class_name, iou in zip(self.config.classes, iou_per_class):
            iou_value[class_name] = iou
        print(iou_value)

        self.metrics_val.reset()
        loss = torch.stack([x for x in outputs]).mean()
        log_dict = {"val_loss": loss, 'val_mIoU': mIoU, 'val_F1': F1, 'val_OA': OA}
        self.log_dict(log_dict, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.config.optimizer
        #scheduler = self.config.scheduler

        return optimizer

    def train_dataloader(self):

        return self.config.train_loader

    def val_dataloader(self):

        return self.config.val_loader


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    #seed_everything(13)

    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=config.weights_path,
                                          filename=config.weights_name)
    
    logger = CSVLogger('lightning_logs', name=config.log_name)
    
    model = Supervision_Train(config)
    
    if config.wandblogger:
        wandblogger = WandbLogger(name=config.wandb_logging_name, project=config.wandb_project)
        wandblogger.watch(model, log_graph=False)
        if config.contrastive:
            if rank_zero_only.rank == 0:
                wandblogger.experiment.config.update(config.config)
        
        logger = [logger]
        logger.append(wandblogger)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='gpu',
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         logger=logger, strategy=config.strategy, callbacks=[checkpoint_callback],
                         accumulate_grad_batches=config.accumulate_grad_batches, 
                         gradient_clip_val=config.gradient_clip_val)
                         #callbacks=[checkpoint_callback], logger=logger) # gradient_clip_val=config.gradient_clip_val,
    
    trainer.fit(model=model)


if __name__ == "__main__":
   main()
