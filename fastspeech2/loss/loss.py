import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.train_config = train_config

    def normalize(self, x, mean, std):
        return (x - mean)/std

    def forward(self, mel, duration_predicted, energy_predicted,
                pitch_predicted, mel_target, duration_predictor_target,
                energy_predictor_target, pitch_predictor_target):
        mel_loss = self.l1_loss(mel, mel_target)
        duration_predictor_loss = self.mse_loss(
            self.normalize(
                duration_predicted.squeeze(),
                self.train_config.alignment_mean, self.train_config.
                alignment_std),
            self.normalize(
                duration_predictor_target.squeeze().float(),
                self.train_config.alignment_mean, self.train_config.
                alignment_std))

        energy_predictor_loss = self.mse_loss(
            energy_predicted.squeeze(),
            energy_predictor_target.squeeze().float())

        pitch_predictor_loss = self.mse_loss(
            pitch_predicted.squeeze(),
            pitch_predictor_target.squeeze().float())

        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss
