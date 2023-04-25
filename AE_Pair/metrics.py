from torch import nn, log10, empty, Tensor

from typing import Dict, Union, List

class PSNR(nn.Module):

    def __init__(self, max_val: int = 1.0) -> None:
        super().__init__()

        self.mse = nn.MSELoss()
        self.log_max_error = 20*log10(max_val)
    
    def forward(self, y: Tensor, y_hat: Tensor):
        loss = self.mse(y, y_hat)

        psnr = self.log_max_error - 10*log10(loss)

        return psnr
    
class MultiMSELoss(nn.Module):

    def __init__(self, weights: Union[Tensor, str, None] = "sum") -> None:
        super().__init__()

        self.mse = nn.MSELoss()

        self.weights = weights

    def forward(self, y: List[Tensor], y_hat: List[Tensor]):
        losses = empty((len(y),))

        for i in range(len(y)):
            losses[i] = self.mse(y[i], y_hat[i])

        if self.weights == "sum":
            return losses.sum(), losses
        elif self.weights == "average":
            return losses.mean(), losses
        else:
            return losses.dot(self.weights.transpose()), losses

