import lightning.pytorch as pl
from torchvision import transforms
from training_loops import AETraining, AEPairedTraining
from AE_Pair.Autoencoders import SimpleAE, SimplePairedAE, SwinAE, SwinPairedAE
from datasets import CIFAR
from torch import nn
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

BATCH_SIZE = 32
HIDDEN_DIM = 24
EPOCHS = 25

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215827, 0.44653124], [0.24703233, 0.24348505, 0.26158768]),
    ]
    )
    dataset = CIFAR(batch_size=BATCH_SIZE, num_workers=0, transform=transform)
    # model = SimplePairedAE(hidden_dim=HIDDEN_DIM, depth=3, activation=nn.ReLU)
    model = SwinPairedAE(hidden_dim=HIDDEN_DIM, depths=[4,4,6], num_heads=[4,4,8], window_size=[2,2])
    # model_part2 = SimpleAE(hidden_dim=HIDDEN_DIM, depth=3, activation=nn.ReLU)
    model_part2 = SwinAE(hidden_dim=HIDDEN_DIM, depths=[4,4,6], num_heads=[4,4,8], window_size=[2,2])
    print(model)
    model = AEPairedTraining(model, lr=1e-3)

    trainer = pl.Trainer(max_epochs=EPOCHS)
    trainer.fit(model, dataset.trainloader, dataset.validloader)

    model_part2.load_state_dict(model.autoencoder.state_dict())
    model = AETraining(model_part2, lr=1e-4)

    trainer = pl.Trainer(default_root_dir=trainer.default_root_dir, max_epochs=EPOCHS)
    trainer.fit(model, dataset.trainloader, dataset.validloader)