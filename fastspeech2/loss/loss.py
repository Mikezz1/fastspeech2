import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(
            self,
            mel, duration_predicted, energy_predicted, mel_target,
            duration_predictor_target,
            energy_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        # print(duration_predicted.squeeze().size(),
        #       duration_predictor_target.squeeze().float().size())
        duration_predictor_loss = self.l1_loss(
            duration_predicted.squeeze(),
            duration_predictor_target.squeeze().float())

        # print(energy_predicted.squeeze().size(),
        #       energy_predictor_target.squeeze().float().size())
        energy_predictor_loss = self.mse_loss(
            energy_predicted.squeeze(),
            energy_predictor_target.squeeze().float())

        return mel_loss, duration_predictor_loss, energy_predictor_loss
