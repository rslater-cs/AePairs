import lightning.pytorch as pl
from AE_Pair.metrics import MultiMSELoss
from torch import Tensor, nn
from torch.optim import Adam

class AETraining(pl.LightningModule):
    def __init__(self, autoencoder: nn.Module) -> None:
        super().__init__()

        self.autoencoder = autoencoder
        self.loss_func = MultiMSELoss(weights='sum')
        self.valid_loss_func = nn.MSELoss()

        self.level_names = []
        for i in range(self.autoencoder.depth):
            self.level_names.append(f'l{i+1}')

    def training_step(self, batch, batch_idx) -> Tensor:
        self.autoencoder.zero_grad()

        x, _= batch

        z_values, z_hat_values = self.autoencoder(x)

        loss, loss_levels = self.loss_func(z_values, z_hat_values)


        data = dict(zip(self.level_names, loss_levels))
        self.log_dict(data)
        self.log("sum_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx) -> Tensor:
        x, _ = batch
        y = x.detach().clone()

        x_hat = self.autoencoder.valid_forward(x)

        loss = self.valid_loss_func(y, x_hat)

        self.log("validation_loss", loss)

        return loss
    
    def configure_optimizers(self):
        optimiser = Adam(self.parameters(), lr=1e-4)
        return optimiser
