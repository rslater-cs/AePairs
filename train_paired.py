import lightning.pytorch as pl
from torchvision import transforms
from training_loops import AETraining, AEPairedTraining
from AE_Pair.networks import SimpleAE, SimplePairedAE, SwinAE, SwinPairedAE
from datasets import CIFAR, IN
from torch import nn
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 64
HIDDEN_DIM = 48
PARIED_EPOCHS = 5
FULL_EPOCHS = 10

if __name__ == '__main__':
    # dataset = IN(batch_size=BATCH_SIZE, num_workers=0)
    dataset = CIFAR(batch_size=BATCH_SIZE, num_workers=0)
    # model = SimplePairedAE(hidden_dim=HIDDEN_DIM, depth=3, activation=nn.ReLU)
    model = SimplePairedAE(hidden_dim=HIDDEN_DIM, depth=3, transfer_dim=16)
    print(model)
    model = AEPairedTraining(model, lr=5e-4)

    trainer = pl.Trainer(max_epochs=PARIED_EPOCHS)
    trainer.fit(model, dataset.trainloader, dataset.validloader)

    model = model.autoencoder.to_autoencoder()
    model = AETraining(model, lr=1e-4)

    trainer = pl.Trainer(default_root_dir=trainer.default_root_dir, max_epochs=FULL_EPOCHS)
    trainer.fit(model, dataset.trainloader, dataset.validloader)