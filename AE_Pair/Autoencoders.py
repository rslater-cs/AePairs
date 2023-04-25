from torch import nn, Tensor
from typing import Tuple

class Permute(nn.Module):

    def __init__(self, dims: Tuple[int]) -> None:
        super().__init__()

        self.dims = dims

    def forward(self, x: Tensor):
        return x.permute(self.dims)
    
class TransformBlock(nn.Module):
    
    def __init__(self, dim: int, activation: nn.Module = nn.ReLU, dropout: float = 0.5) -> None:
        super().__init__()

        self.transform = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: Tensor):
        x =  self.transform(x)
        return x

class SimpleEncoderBlock(nn.Module):

    def __init__(self, dim: int, activation: nn.Module = nn.ReLU, dropout: float = 0.5) -> None:
        super().__init__()

        self.transform = TransformBlock(dim, activation, dropout=dropout)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1),
            activation(),
        )

        self.norm = nn.LayerNorm(dim)

        self.normalise = nn.Sequential(
            Permute((0, 2, 3, 1)),
            self.norm,
            Permute((0, 3, 1, 2)),
            nn.Dropout2d(dropout)
        )

    def forward(self, x: Tensor):
        x = self.transform(x)
        x = self.reduce(x)
        x = self.normalise(x)

        return x
    
class SimpleDecoderBlock(nn.Module):

    def __init__(self, dim: int, activation: nn.Module = nn.ReLU, dropout: float = 0.5) -> None:
        super().__init__()

        self.transform = TransformBlock(dim, activation, dropout=dropout)

        self.enlarge = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1, dilation=1),
            activation(),
        )

        self.norm = nn.LayerNorm(dim)

        self.normalise = nn.Sequential(
            Permute((0, 2, 3, 1)),
            self.norm,
            Permute((0, 3, 1, 2)),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: Tensor):
        x = self.transform(x)
        x = self.enlarge(x)
        x = self.normalise(x)

        return x


class AE(nn.Module):

    def __init__(self, hidden_dim: int, depth: int, encoder_block: nn.Module = SimpleEncoderBlock, decoder_block: nn.Module = SimpleDecoderBlock, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(encoder_block(dim=hidden_dim, dropout=dropout, activation=activation))
            self.decoders.append(decoder_block(dim=hidden_dim, dropout=dropout, activation=activation))
        
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=1, stride=1),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2)),
            nn.Dropout2d(dropout),
        )

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=3, kernel_size=1, stride=1),
            # nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        z = self.encoders[0](self.embed(x))

        for i in range(1, self.depth):
            z = self.encoders[i](z)
        
        for i in range(self.depth-1, 0, -1):
            z = self.decoders[i](z)
        
        x_hat = self.head(self.decoders[0](z))

        return x_hat
    
    def valid_forward(self, x: Tensor):
        return self.forward(x)
    

class AE_Paired(AE):
    def __init__(self, hidden_dim: int, depth: int, encoder_block: nn.Module = SimpleEncoderBlock, decoder_block: nn.Module = SimpleDecoderBlock, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        super().__init__(hidden_dim, depth, encoder_block, decoder_block, dropout, activation)    

        self.latent = [None]*depth
        self.reconstructed_latent = [None]*depth

    def forward(self, x: Tensor):
        self.latent[0] = x.detach().clone()

        z = self.encoders[0](self.embed(x))
        self.reconstructed_latent[0] = self.head(self.decoders[0](z))

        for i in range(1, self.depth):
            self.latent[i] = z.detach().clone()
            z = self.encoders[i](self.latent[i])
            self.reconstructed_latent[i] = self.decoders[i](z)

        return self.latent, self.reconstructed_latent
    
    def valid_forward(self, x: Tensor):
        z = self.encoders[0](self.embed(x))
        for i in range(1, self.depth):
            z = self.encoders[i](z)

        for i in range(self.depth-1, -1, -1):
            z = self.decoders[i](z)

        return self.head(z)


class SimplePairedAE(AE_Paired):

    def __init__(self, hidden_dim: int, depth: int, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        super().__init__(hidden_dim, depth, SimpleEncoderBlock, SimpleDecoderBlock, dropout, activation)


class SimpleAE(AE):
    def __init__(self, hidden_dim: int, depth: int, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        super().__init__(hidden_dim, depth, SimpleEncoderBlock, SimpleDecoderBlock, dropout, activation)
