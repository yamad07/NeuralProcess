from src.trainers.neural_process_trainer import NeuralProcessTrainer
from src.models.neural_process import NeuralProcess

from torchvision.datasets import MNIST
from torchvision import transforms
import torch.utils.data as data


def main(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset = MNIST('./data/', transform=transform, download=True)
    data_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    neural_process = NeuralProcess()
    trainer = NeuralProcessTrainer(neural_process=neural_process, data_loader=data_loader)
    trainer.train(n_epoch=args.n_epoch)

if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig(format='[Neural Process] %(levelname)s: %(message)s',
                        level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    main(args)
