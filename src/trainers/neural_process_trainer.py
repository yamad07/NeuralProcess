import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class NeuralProcessTrainer(object):

    def __init__(self, neural_process, data_loader, context_size=300):
        self.neural_process = neural_process
        self.data_loader = data_loader
        self.context_size = context_size

        self.criterion = nn.MSELoss()
        self.optim = optim.Adam(self.neural_process.parameters(), lr=1e-3)
        self.device = torch.device('cpu')

    def train(self, n_epoch):

        self.neural_process.to(self.device)
        for epoch in range(n_epoch):
            for i_of_data, (minibatch_img, labels) in enumerate(self.data_loader):
                batch_size = minibatch_img.size(0)
                for i_of_minibatch in range(batch_size):

                    img = minibatch_img[i_of_minibatch]
                    self.neural_process.train()

                    img_size = img.size(1)
                    n_pixels = img_size ** 2

                    all_x = torch.stack([torch.Tensor([i, j]) for i in range(img_size) for j in range(img_size)])
                    all_y = img.view(-1, 1)
                    indexes = torch.randperm(n_pixels)
                    context_indexes = indexes[:self.context_size]
                    target_indexes = indexes[self.context_size:]

                    context_x = all_x[context_indexes]
                    context_y = all_y[context_indexes]
                    target_x = all_x[target_indexes]
                    target_y = all_y[target_indexes]

                    context_x = context_x.to(self.device)
                    context_y = context_y.to(self.device)

                    for i_of_update in range(50):
                        self.optim.zero_grad()
                        target_logits, context_logits, context_preds_logits, target_preds_logits, kl_divergence = self.neural_process(context_x, context_y, target_x, target_y)
                        loss = F.binary_cross_entropy(target_logits, target_y) + kl_divergence
                        loss.backward()
                        self.optim.step()

                    logger.info('epoch: {} loss: {} KL: {}'.format(epoch, loss, kl_divergence))

                    target_img = torch.zeros(n_pixels)
                    target_img[target_indexes] = target_y.squeeze()
                    target_img[context_indexes] = context_y.squeeze()
                    target_img = target_img.view(img_size, img_size)

                    context_img = torch.zeros(n_pixels)
                    context_img[context_indexes] = context_y.squeeze()
                    context_img = context_img.view(img_size, img_size)

                    all_preds_img = torch.zeros(n_pixels)
                    all_preds_img[target_indexes] = target_preds_logits.squeeze()
                    all_preds_img[context_indexes] = context_preds_logits.squeeze()
                    all_preds_img = all_preds_img.view(img_size, img_size)

                    img = torch.cat((context_img, target_img, all_preds_img), dim=0)
                    vutils.save_image(img, 'Epoch_{}_data.jpg'.format(epoch, i_of_data), normalize=True)
