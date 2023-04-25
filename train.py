import lightning.pytorch as pl
from training_loops import AETraining
from AE_Pair.Autoencoders import SimpleAE
from datasets import CIFAR

BATCH_SIZE = 32
HIDDEN_DIM = 48

if __name__ == '__main__':
    dataset = CIFAR(batch_size=BATCH_SIZE, num_workers=0)
    model = SimpleAE(hidden_dim=48, depth=3)
    print(model)
    model = AETraining(model)

    trainer = pl.Trainer(max_epochs=5)
    trainer.fit(model, dataset.trainloader, dataset.validloader)