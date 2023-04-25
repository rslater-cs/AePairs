import lightning.pytorch as pl
from torchvision import transforms
from training_loops import AETraining
from AE_Pair.Autoencoders import SimpleAE
from datasets import CIFAR
from torch import nn

BATCH_SIZE = 8
HIDDEN_DIM = 48

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
    ]
    )
    dataset = CIFAR(batch_size=BATCH_SIZE, num_workers=0, transform=transform)
    model = SimpleAE(hidden_dim=48, depth=3, activation=nn.LeakyReLU)
    print(model)
    model = AETraining(model)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, dataset.trainloader, dataset.validloader)