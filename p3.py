import pandas as pd
import torch
from torch import nn, optim
import numpy as np
from typing import Any, Union, List
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from itertools import product
import os, json
from sklearn.metrics import mean_squared_error
import wandb
from copy import deepcopy

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
BATCH_SIZE = 1024
SEED = 26
torch.random.manual_seed(SEED)
np.random.seed(SEED)
torch.set_float32_matmul_precision('high')

ACT_FN_NAMEs = ('ReLU', 'Sigmoid', 'Tanh', 'ELU', 'GELU',)
OPT_NAMEs = ('Adam', 'AdamW', 'SGD',)
"""DIMs = (
    [16, 16],
    [32, 32],
    [64, 64],
)"""
DIMs = (
    [8,],
    [16,],
    [32,],
    [64,],
    [8, 8,],
    [16, 16],
    [32, 32],
    [64, 64],
    [8, 8, 8],
    [16, 16, 16],
    [32, 32, 32],
    [64, 64, 64],
)
def standardize(a: np.ndarray, mu: Union[np.ndarray, float], sigma: Union[np.ndarray, float]):
    return (a - mu) / sigma

def un_standardize(b: np.ndarray, mu: Union[np.ndarray, float], sigma: Union[np.ndarray, float]):
    return b * sigma + mu


class RegDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        assert self.x.shape[0] == self.y.shape[0]
        
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    
class RegMLP(pl.LightningModule):
    def __init__(
        self,
        opt_name: str='AdamW',
        lr: float=1e-3,
        act_fn_name: str='ReLU',
        dims: List[int]=[32, 32]
    ):
        super().__init__()
        if act_fn_name == 'ReLU':
            act_fn = nn.ReLU
        elif act_fn_name == 'Sigmoid':
            act_fn = nn.Sigmoid
        elif act_fn_name == 'Tanh':
            act_fn = nn.Tanh
        elif act_fn_name == 'ELU':
            act_fn = nn.ELU
        elif act_fn_name == 'GELU':
            act_fn = nn.GELU
        else:
            raise NotImplementedError('this activation function is not supported......')
            
        layers = []
        for i, dim in enumerate(dims):
            layers.append(nn.Linear(18, dim) if i == 0 else nn.Linear(dims[i-1], dim))
            layers.append(act_fn())
        layers.append(nn.Linear(dims[-1], 1))
        
        self.model = nn.Sequential(*layers)
        self.lr = lr
        self.opt_name = opt_name

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return outputs
    
    def configure_optimizers(self):
        if self.opt_name == 'AdamW':
            opt_cls = optim.AdamW
        elif self.opt_name == 'Adam':
            opt_cls = optim.Adam
        elif self.opt_name == 'SGD':
            opt_cls = optim.SGD
        else:
            raise NotImplementedError('this optimizer is not supported......')
        return opt_cls(self.parameters(), lr=self.lr)
    
    def _step(self, batch):
        x, y = batch
        y_pred = self.forward(x).flatten()
        loss_func = nn.MSELoss()
        loss = loss_func(y_pred, y)
        return loss
    
    def training_step(self, train_batch, *args: Any, **kwargs: Any):
        train_loss = self._step(train_batch)
        self.log('train_loss', train_loss)
        return train_loss
    
    def validation_step(self, val_batch, *args: Any, **kwargs: Any):
        val_loss = self._step(val_batch)
        self.log('val_loss', val_loss)
        
    
if __name__ == '__main__':
    train_data = pd.read_csv('Reg_Train.txt', sep=' ', header=None)
    test_data = pd.read_csv('Reg_Test.txt', sep=' ', header=None)

    train_x = np.array(train_data.iloc[:, :-1])
    train_y = np.array(train_data.iloc[:, -1])
    test_x = np.array(train_data.iloc[:, :-1])
    test_y = np.array(train_data.iloc[:, -1])

    x_mean = np.mean(train_x, axis=0, keepdims=True)
    x_std = np.std(train_x, axis=0, keepdims=True)
    y_mean = np.mean(train_y)
    y_std = np.std(train_y)
    
    train_set = RegDataset(
        x=standardize(train_x, x_mean, x_std),
        ### y=standardize(train_y, y_mean, y_std),
        y=train_y
    )
    train_set, val_set = random_split(train_set, (0.9, 0.1))
    test_set = RegDataset(
        x=standardize(test_x, x_mean, x_std),
        ### y=standardize(test_y, y_mean, y_std),
        y=test_y
    )
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
    
    # print(next(iter(train_loader)))
    results = []
    print("Total number of runs:", len(ACT_FN_NAMEs)*len(OPT_NAMEs)*len(DIMs))
    for act_fn_name, opt_name, dims in product(ACT_FN_NAMEs, OPT_NAMEs, DIMs):
        config = {
            'activation_functiom': act_fn_name,
            'opt_name': opt_name,
            'hidden_dimensions': '-'.join([str(_) for _ in dims])
        }
        
        model = RegMLP()
        run = wandb.init(
            entity='gariscat',
            project='AI701_Assignment2_Problem3_standardize_x',
            config=config,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices="auto",
            logger=WandbLogger(),
            max_epochs=100,
            deterministic=True,
            default_root_dir=LOG_DIR,
            log_every_n_steps=100,
        )
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        
        test_y_pred = model.forward(torch.tensor(standardize(test_x, x_mean, x_std), dtype=torch.float32))
        test_y_pred = test_y_pred.detach().cpu().flatten().numpy()
        ### test_y_pred = un_standardize(test_y_pred, y_mean, y_std)
        # error = mean_squared_error(test_y, test_y_pred, multioutput='raw_values')
        error = (test_y - test_y_pred) ** 2
        l1_dis = np.abs(test_y - test_y_pred)
        # result = deepcopy(config)
        # result.update({'error_avg': np.mean(error), 'error_std': np.std(error)})
        # results.append(str(result)+'\n')
        run.log({'error_avg': np.mean(error)})
        run.log({'error_std': np.std(error)})
        run.log({'l1_avg': np.mean(l1_dis)})
        run.log({'l1_std': np.std(l1_dis)})
        run.finish()
    
    
    """with open('p3_results_standardize_x.txt', 'w') as f:
        f.writelines(results)"""
        