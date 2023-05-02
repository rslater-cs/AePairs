from torch import nn, Tensor, concat
import torch
from typing import Tuple, Dict, Any, List, Callable, Union
from torchvision.models.swin_transformer import SwinTransformerBlock
from torchvision.models.swin_transformer import PatchMerging as SwinPatchMerging

class NoneLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x

class Permute(nn.Module):

    def __init__(self, dims: Tuple[int]) -> None:
        super().__init__()

        self.dims = dims

    def forward(self, x: Tensor):
        return x.permute(self.dims)
    
class PatchSplitting(nn.Module):

    def __init__(self, dim, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.enlargement = nn.Linear(dim, 4*dim)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        device = x.get_device()
        device = 'cpu' if device < 0 else device 

        x = self.enlargement(x) # B H W 2C

        x_s = torch.split(x, split_size_or_sections=C, dim=3)

        # device = "cuda:0" if torch.cuda.is_available() else "cpu"

        x = torch.empty(B, 2*H, 2*W, C).to(device)

        x[:, 0::2, 0::2, :] = x_s[0]
        x[:, 1::2, 0::2, :] = x_s[1]
        x[:, 0::2, 1::2, :] = x_s[2]
        x[:, 1::2, 1::2, :] = x_s[3]

        x = self.norm(x) # B H W 2C

        return x
    
class PatchMerging(SwinPatchMerging):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__(dim, norm_layer)
        self.reduction = nn.Linear(4*dim, dim, bias=False)
    
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

        self.aggregate = nn.Conv2d(in_channels=2*dim, out_channels=dim, kernel_size=1, stride=1)

    def forward(self, x: Tensor):
        z = self.transform(x)
        x = self.aggregate(concat((z, x), dim=1))
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
    

class SwinEncoderBlock(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: List[int], mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0) -> None:
        super().__init__()

        blocks = []

        for i_layer in range(depth):   
            blocks.append(SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout 
            ))
        self.transform = nn.Sequential(*blocks)

        self.reduce = PatchMerging(dim)

    def forward(self, x: Tensor):
        x = x.permute((0,2,3,1))
        x = self.transform(x)
        x = self.reduce(x)
        x = x.permute((0, 3, 1, 2))
        return x
    
class SwinDecoderBlock(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: List[int], mlp_ratio: float = 4.0, dropout: float = 0.0, attention_dropout: float = 0.0) -> None:
        super().__init__()

        blocks = []

        for i_layer in range(depth):   
            blocks.append(SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout 
            ))
        self.transform = nn.Sequential(*blocks)

        self.enlarge = PatchSplitting(dim)

    def forward(self, x: Tensor):
        x = x.permute((0, 2, 3, 1))
        x = self.transform(x)
        x = self.enlarge(x)
        x = x.permute((0, 3, 1, 2))
        return x


class AE(nn.Module):

    def __init__(self, hidden_dim: int, depth: int, transfer_dim: int = -1, encoder_block: nn.Module = SimpleEncoderBlock, decoder_block: nn.Module = SimpleDecoderBlock, dropout: float = 0.5, activation: nn.Module = nn.ReLU, block_args: List[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth):
            self.encoders.append(encoder_block(**block_args[i]))
            self.decoders.append(decoder_block(**block_args[i]))
        
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=1, stride=1),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2)),
            nn.Dropout2d(dropout),
        )

        if transfer_dim < 0:
            self.to_transfer = NoneLayer()
            self.from_transfer = NoneLayer()
        else:
            self.to_transfer = nn.Conv2d(hidden_dim, transfer_dim, kernel_size=1, stride=1)
            self.from_transfer = nn.Conv2d(transfer_dim, hidden_dim, kernel_size=1, stride=1)

        self.head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            activation(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim//2, kernel_size=3, stride=1, padding=1),
            activation(),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim//2),
            Permute((0, 3, 1, 2)),
            nn.Conv2d(in_channels=hidden_dim//2, out_channels=3, kernel_size=1, stride=1),
            # nn.Sigmoid(),
        )

    def forward(self, x: Tensor):
        z = self.encoders[0](self.embed(x))

        for i in range(1, self.depth):
            z = self.encoders[i](z)
        
        z = self.to_transfer(z)
        z = self.from_transfer(z)
        
        for i in range(self.depth-1, 0, -1):
            z = self.decoders[i](z)
        
        x_hat = self.head(self.decoders[0](z))

        return x_hat
    
    def valid_forward(self, x: Tensor):
        return self.forward(x)
    

class AE_Paired(AE):
    def __init__(self, hidden_dim: int, depth: int, transfer_dim: int = -1, encoder_block: nn.Module = SimpleEncoderBlock, decoder_block: nn.Module = SimpleDecoderBlock, dropout: float = 0.5, activation: nn.Module = nn.ReLU, block_args: List[Dict[str, Any]] = None) -> None:
        super().__init__(hidden_dim, depth, transfer_dim, encoder_block, decoder_block, dropout, activation, block_args)    

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

    def __init__(self, hidden_dim: int, depth: int, transfer_dim: int = -1, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        block_args = dict()
        block_args['dropout'] = dropout
        block_args['activation'] = activation
        block_args['dim'] = hidden_dim

        block_args_list = []
        for i in range(depth):
            block_args_list.append(block_args)

        super().__init__(hidden_dim, depth, transfer_dim, SimpleEncoderBlock, SimpleDecoderBlock, dropout, activation, block_args_list)


class SimpleAE(AE):
    def __init__(self, hidden_dim: int, depth: int, transfer_dim: int = -1, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        block_args = dict()
        block_args['dropout'] = dropout
        block_args['activation'] = activation
        block_args['dim'] = hidden_dim

        block_args_list = []
        for i in range(depth):
            block_args_list.append(block_args)
        
        super().__init__(hidden_dim, depth, transfer_dim, SimpleEncoderBlock, SimpleDecoderBlock, dropout, activation, block_args_list)

class SwinAE(AE):
    def __init__(self, hidden_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], transfer_dim: int = -1, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0, activation: nn.Module = nn.ReLU) -> None:
        block_args = []

        for i in range(len(depths)):
            block_arg = dict()
            block_arg['dim'] = hidden_dim
            block_arg['depth'] = depths[i]
            block_arg['num_heads'] = num_heads[i]
            block_arg['window_size'] = window_size
            block_arg['mlp_ratio'] = mlp_ratio
            block_arg['dropout'] = dropout
            block_arg['attention_dropout'] = attn_dropout
            block_args.append(block_arg)

        super().__init__(hidden_dim, len(depths), transfer_dim, SwinEncoderBlock, SwinDecoderBlock, dropout, activation, block_args)

class SwinPairedAE(AE_Paired):
    def __init__(self, hidden_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], transfer_dim: int = -1, mlp_ratio: float = 4.0, dropout: float = 0.0, attn_dropout: float = 0.0, activation: nn.Module = nn.ReLU) -> None:
        block_args = []

        for i in range(len(depths)):
            block_arg = dict()
            block_arg['dim'] = hidden_dim
            block_arg['depth'] = depths[i]
            block_arg['num_heads'] = num_heads[i]
            block_arg['window_size'] = window_size
            block_arg['mlp_ratio'] = mlp_ratio
            block_arg['dropout'] = dropout
            block_arg['attention_dropout'] = attn_dropout
            block_args.append(block_arg)

        super().__init__(hidden_dim, len(depths), transfer_dim, SwinEncoderBlock, SwinDecoderBlock, dropout, activation, block_args)
