import lightning.pytorch as pl
from torchvision import transforms
from training_loops import AETraining, AEPairedTraining
from AE_Pair.Autoencoders import SimpleAE, SimplePairedAE
from datasets import CIFAR
from torch import nn
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 32
HIDDEN_DIM = 48

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
    ]
    )
    dataset = CIFAR(batch_size=BATCH_SIZE, num_workers=0, transform=transform)
    model = SimplePairedAE(hidden_dim=48, depth=3, activation=nn.LeakyReLU)
    print(model)
    model = AEPairedTraining(model)

    trainer = pl.Trainer(max_epochs=20)
    trainer.fit(model, dataset.trainloader, dataset.validloader)