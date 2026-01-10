import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, CohenKappa

class LitEEGModel(pl.LightningModule):
    def __init__(self, net: nn.Module, sr_layer: nn.Module = None, lr: float = 0.001, weight_decay: float = 0.01,
                 model_name: str = "", datamodule_name: str = "", mechanism_name: str = "", noise_name: str = ""):
        super().__init__()
        # 保存所有超参数，包括用于标识实验的元数据
        # 注意: 参数名使用 datamodule_name 而非 dataset_name，以避免与 DataModule 的 hparams 冲突
        self.save_hyperparameters(ignore=['net', 'sr_layer'])
        
        self.net = net
        self.sr_layer = sr_layer
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        # Assuming 4 classes for BCI IV 2a as default, but should be configurable.
        # We can infer from net.nb_classes or net.n_classes if available, or pass as arg.
        # For now, we'll try to inspect the network or default to 4.
        n_classes = getattr(net, 'nb_classes', getattr(net, 'n_classes', 4))
        
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=n_classes)
        
        self.val_f1 = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        self.test_f1 = F1Score(task="multiclass", num_classes=n_classes, average='macro')
        
        self.val_kappa = CohenKappa(task="multiclass", num_classes=n_classes)
        self.test_kappa = CohenKappa(task="multiclass", num_classes=n_classes)

    def forward(self, x):
        if self.sr_layer is not None:
            x = self.sr_layer(x)
        return self.net(x)

    def training_step(self, batch, batch_idx):
        # Braindecode WindowsDataset returns (x, y, ind)
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_kappa(preds, y)
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        self.log("val_kappa", self.val_kappa, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.test_f1(preds, y)
        self.test_kappa(preds, y)
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        self.log("test_kappa", self.test_kappa, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }