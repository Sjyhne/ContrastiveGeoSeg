import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset

class ParameterNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.trainable_param = nn.init.uniform_(nn.Parameter(torch.tensor(0.), requires_grad=True))
        
    def forward(self, x):
        return x + self.trainable_param

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.param_net = ParameterNetwork()
        
        self.net = nn.Sequential(*[
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, 2),
        ])
        
        self.nets = nn.ModuleList([
            nn.Sequential(*[
            nn.Linear(100, 100),
            nn.GELU(),
            nn.Linear(100, 100),
            ]) for i in range(4)
        ])
    
    def forward(self, x):
        
        for i in range(4):
            x = self.nets[i](x)
            x = self.param_net(x)        
        
        x = self.net(x)
        
        return x
        
def get_datasets(n_train=200, n_valid=200,
                 input_shape=[100, 100], target_shape=[],
                 n_classes=None):
    """Construct and return random number datasets"""
    train_x = torch.randn([n_train] + input_shape)
    valid_x = torch.randn([n_valid] + input_shape)
    if n_classes is not None:
        train_y = torch.randint(n_classes, [n_train] + target_shape, dtype=torch.long)
        valid_y = torch.randint(n_classes, [n_valid] + target_shape, dtype=torch.long)
    else:
        train_y = torch.randn([n_train] + target_shape)
        valid_y = torch.randn([n_valid] + target_shape)
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)
    return train_dataset, valid_dataset, {}
        

class Reproduce(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = nn.MSELoss()
    
    def training_step(self, batch, batch_idx):
        input, label = batch
        
        out = self.model(input)
        
        loss = self.loss(out, label)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)
        return optimizer
    
    def train_dataloader(self):
        ds, _, _ = get_datasets()
        ds = torch.utils.data.DataLoader(dataset=ds, batch_size=2, pin_memory=True)
        return ds


model = Model()
model = Reproduce(model)
trainer = pl.Trainer(devices=[14, 15], accelerator="gpu", strategy="ddp")
trainer.fit(model=model)

    
    