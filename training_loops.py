import lightning.pytorch as pl
from AE_Pair.metrics import MultiMSELoss
from AE_Pair.networks import AE, AE_Paired
from torch import Tensor, nn, log10
from torch.optim import Adam

class AETraining(pl.LightningModule):
    def __init__(self, autoencoder: AE, lr: float = 1e-3) -> None:
        super().__init__()

        self.autoencoder = autoencoder
        self.loss_func = nn.MSELoss()
        self.lr = lr

    def training_step(self, batch, batch_idx) -> Tensor:
        self.autoencoder.zero_grad()

        x, _= batch
        y = x.detach().clone()

        x_hat = self.autoencoder(x)

        loss = self.loss_func(x_hat, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        x, _ = batch
        y = x.detach().clone()

        x_hat = self.autoencoder(x)

        loss = self.loss_func(y, x_hat)
        psnr = -10*log10(loss)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)

        return loss
    
    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=self.lr)
        return optimiser
    
class AEPairedTraining(pl.LightningModule):

    def __init__(self, autoencoder: AE_Paired, lr: float = 1e-3) -> None:
        super().__init__()
        self.autoencoder = autoencoder

        self.lr = lr

        # self.loss_func = MultiMSELoss(weights='sum')
        self.loss_func = nn.MSELoss()

        self.level_names = []
        for i in range(len(self.autoencoder.pairs)):
            self.level_names.append(f'l{i+1}')

    def training_step(self, batch, batch_idx) -> Tensor:
        self.autoencoder.zero_grad()

        x, _= batch

        losses = self.autoencoder(x)

        loss = losses.sum()

        data = dict(zip(self.level_names, losses))
        self.log_dict(data, on_step=False, on_epoch=True)
        self.log("sum_loss", loss, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        if batch_idx == 0:
            self.autoencoder_val = self.autoencoder.to_autoencoder()

        x, _ = batch
        y = x.detach().clone()

        x_hat = self.autoencoder_val(x)

        loss = self.loss_func(y, x_hat)
        psnr = -10*log10(loss)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=self.lr)
        return optimiser
