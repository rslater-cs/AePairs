import lightning.pytorch as pl
from torchvision import transforms
from training_loops import AETraining
from AE_Pair.Autoencoders import SimpleAE, SwinAE
from datasets import CIFAR
from torch import nn
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 32
HIDDEN_DIM = 192
EPOCHS = 20

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
    ]
    )
    dataset = CIFAR(batch_size=BATCH_SIZE, num_workers=0, transform=transform)
    # model = SimpleAE(hidden_dim=HIDDEN_DIM, depth=3, activation=nn.ReLU)
    model = SwinAE(hidden_dim=HIDDEN_DIM, depths=[4,4,6], num_heads=[4,4,8], window_size=[2,2])
    print(model)
    model = AETraining(model, lr=5e-4)

    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(model, dataset.trainloader, dataset.validloader)

    model = AETraining(model.autoencoder, lr=1e-4)

    trainer = pl.Trainer(default_root_dir=trainer.default_root_dir, max_epochs=EPOCHS)
    trainer.fit(model, dataset.trainloader, dataset.validloader)