import torch
from torch import nn
from torchvision.models.swin_transformer import PatchMerging as SwinPatchMerging
from torchvision.models.swin_transformer import SwinTransformerBlock

from typing import Callable, Tuple, Union, List, Dict, Any

class Permute(nn.Module):

    def __init__(self, dims: Tuple[int]) -> None:
        super().__init__()

        self.dims = dims

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        z = self.transform(x)
        x = self.aggregate(torch.concat((z, x), dim=1))
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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        x = x.permute((0, 2, 3, 1))
        x = self.transform(x)
        x = self.enlarge(x)
        x = x.permute((0, 3, 1, 2))
        return x
    
class Pair(nn.Module):
    def __init__(self, encoder_block: nn.Module, decoder_block: nn.Module):
        super().__init__()
        self.encoder = encoder_block
        self.decoder = decoder_block
        self.loss_func = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = self.loss_func(x, x_hat)

        return z.detach().clone(), loss
    
class BottleNeck(Pair):
    def __init__(self, to_bottleneck: nn.Module, from_bottleneck: nn.Module):
        super().__init__(to_bottleneck, from_bottleneck)


class AE(nn.Module):

    def __init__(self, hidden_dim: int, depth: int, transfer_block: Union[Tuple[nn.Module, nn.Module], None] = None, encoder_block: nn.Module = SimpleEncoderBlock, decoder_block: nn.Module = SimpleDecoderBlock, dropout: float = 0.5, activation: nn.Module = nn.ReLU, block_args: List[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.depth = depth

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=1, stride=1),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2)),
            nn.Dropout2d(dropout),
        )

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

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.encoders.append(self.embed)
        
        for i in range(depth):
            self.encoders.append(encoder_block(**block_args[i]))
            self.decoders.append(decoder_block(**block_args[i]))

        self.decoders.append(self.head)

        if transfer_block != None:
            self.encoders.append(transfer_block[0])
            self.decoders.insert(0, transfer_block[1])

        self.encoder = nn.Sequential(*self.encoders)
        self.decoder = nn.Sequential(*self.decoders)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    

class AE_Paired(nn.Module):
    def __init__(self, hidden_dim: int, depth: int, transfer_block: Union[Tuple[nn.Module, nn.Module], None] = None, encoder_block: nn.Module = SimpleEncoderBlock, decoder_block: nn.Module = SimpleDecoderBlock, dropout: float = 0.5, activation: nn.Module = nn.ReLU, block_args: List[Dict[str, Any]] = None) -> None:
        super().__init__()

        self.args = {
            "hidden_dim":hidden_dim,
            "depth":depth,
            "transfer_block":transfer_block,
            "encoder_block":encoder_block,
            "decoder_block":decoder_block,
            "dropout":dropout,
            "activation":activation,
            "block_args":block_args
        }   

        self.depth = depth

        self.embed = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=1, stride=1),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2)),
            nn.Dropout2d(dropout),
        )

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

        self.pairs = nn.ModuleList()

        self.first_encoder = encoder_block(**block_args[0])
        self.first_decoder = decoder_block(**block_args[0])

        e = nn.Sequential(self.embed, self.first_encoder)
        d = nn.Sequential(self.first_decoder, self.head)

        self.pairs.append(Pair(e, d))
        
        for i in range(1, depth):
            self.pairs.append(Pair(encoder_block(**block_args[i]), decoder_block(**block_args[i])))

        if transfer_block != None:
            self.pairs.append(BottleNeck(transfer_block[0], transfer_block[1]))

    def forward(self, x: torch.Tensor):
        losses = torch.empty((len(self.pairs),))

        z = x
        for i in range(len(self.pairs)):
            z, losses[i] = self.pairs[i](z)

        return losses
    
    def to_autoencoder(self) -> AE:
        device = next(self.parameters()).device
        autoencoder = AE(**self.args).to(device)

        autoencoder.embed.load_state_dict(self.embed.state_dict())
        autoencoder.head.load_state_dict(self.head.state_dict())

        encoders = nn.ModuleList()
        decoders = nn.ModuleList()

        encoders.append(self.embed)
        encoders.append(self.first_encoder)
        
        for i in range(1, self.depth):
            encoders.append(self.pairs[i].encoder)
            decoders.append(self.pairs[self.depth-i].decoder)

        decoders.append(self.first_decoder)

        decoders.append(self.head)

        if self.args['transfer_block'] != None:
            encoders.append(self.pairs[-1].encoder)
            decoders.insert(0, self.pairs[-1].decoder)

        encoder = nn.Sequential(*encoders)
        decoder = nn.Sequential(*decoders)

        autoencoder.encoder.load_state_dict(encoder.state_dict())
        autoencoder.decoder.load_state_dict(decoder.state_dict())

        return autoencoder
    

class SimplePairedAE(AE_Paired):

    def __init__(self, hidden_dim: int, depth: int, transfer_dim: int = -1, dropout: float = 0.5, activation: nn.Module = nn.ReLU) -> None:
        block_args = dict()
        block_args['dropout'] = dropout
        block_args['activation'] = activation
        block_args['dim'] = hidden_dim

        to_bottle = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=1, stride=1),
            activation(),
            nn.Conv2d(hidden_dim//2, transfer_dim, kernel_size=1, stride=1),
            activation(),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(transfer_dim),
            Permute((0, 3, 1, 2))
        )
        from_bottle = nn.Sequential(
            nn.Conv2d(transfer_dim, hidden_dim//2, kernel_size=1, stride=1),
            activation(),
            nn.Conv2d(hidden_dim//2, hidden_dim, kernel_size=1, stride=1),
            activation(),
            Permute((0, 2, 3, 1)),
            nn.LayerNorm(hidden_dim),
            Permute((0, 3, 1, 2))
        )

        block_args_list = []
        for i in range(depth):
            block_args_list.append(block_args)

        super().__init__(hidden_dim, depth, (to_bottle, from_bottle), SimpleEncoderBlock, SimpleDecoderBlock, dropout, activation, block_args_list)


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
