import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.neural_process import fc_layer

class NeuralProcess(nn.Module):

    def __init__(self, hidden_dim=256):
        super(NeuralProcess, self).__init__()
        self.hidden_dim = 30
        self.h = nn.Sequential(
                fc_layer(3, 400),
                fc_layer(400, 400),
                fc_layer(400, 400),
                )

        self.mean_aggregater = nn.Sequential(
                nn.Linear(400, 400),
                nn.LeakyReLU(),
                nn.Linear(400, self.hidden_dim),
                )
        self.variance_aggregater = nn.Sequential(
                nn.Linear(400, 400),
                nn.LeakyReLU(),
                nn.Linear(400, self.hidden_dim),
                )

        self.g = nn.Sequential(
                fc_layer(2 + self.hidden_dim, 400),
                fc_layer(400, 400),
                fc_layer(400, 400),
                fc_layer(400, 400),
                nn.Linear(400, 1)
                )

    def forward(self, context_x, context_y, target_x, target_y):
        context = torch.cat((context_x, context_y), dim=1)
        context_representations = self.h(context).mean(0)

        context_mean = self.mean_aggregater(context_representations)
        context_std = self.variance_aggregater(context_representations)
        context_var = torch.exp(0.5 * context_std)
        context_z = torch.rand_like(context_var) * context_var + context_mean

        all_x = torch.cat((context_x, target_x), dim=0)
        all_y = torch.cat((context_y, target_y), dim=0)

        all_representations = self.h(torch.cat((all_x, all_y), dim=1)).mean(0)
        all_mean = self.mean_aggregater(all_representations)
        all_std = self.variance_aggregater(all_representations)
        all_var = torch.exp(0.5 * all_std)
        all_z = torch.rand_like(all_var) * all_var + all_mean

        x_domain = all_x.size(0)
        target_x_domain = target_x.size(0)
        context_x_domain = context_x.size(0)

        expanded_all_z = torch.stack([all_z for i in range(target_x_domain)])
        target_logits = F.sigmoid(self.g(
            torch.cat((expanded_all_z, target_x), dim=1)))

        expanded_all_z = torch.stack([all_z for i in range(context_x_domain)])
        context_logits = F.sigmoid(self.g(
            torch.cat((expanded_all_z, context_x), dim=1)))

        expanded_context_z = torch.stack([context_z for i in range(context_x_domain)])
        context_preds_logits = F.sigmoid(self.g(
            torch.cat((expanded_context_z, context_x), dim=1)))

        expanded_target_z = torch.stack([context_z for i in range(target_x_domain)])
        target_preds_logits = F.sigmoid(self.g(
            torch.cat((expanded_target_z, target_x), dim=1)))

        negative_kl_divergence = self._calc_negative_kl_divergence(
                context_mean, context_std, all_mean, all_std).sum()

        return target_logits, context_logits, context_preds_logits, \
            target_preds_logits, negative_kl_divergence

    def _calc_negative_kl_divergence(self, c_mu, c_std, a_mu, a_std):
        c_std = torch.exp(0.5 * c_std)
        a_std = torch.exp(0.5 * a_std)
        return  torch.log(c_std / a_std) + \
            ((a_std ** 2 + (a_mu - c_mu) ** 2) / (2 * c_std ** 2)) - 0.5
