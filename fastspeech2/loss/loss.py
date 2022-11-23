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
                energy_predictor_target, pitch_predictor_target, mel_mask,
                src_mask):

        # mel_mask = mel_mask[:, :mel_mask.shape[1]]
        # print(mel.size())
        # print(mel_mask.size())
        # print(mel_target.size())
        mel = mel.masked_select(mel_mask.unsqueeze(2))
        mel_target = mel_target.masked_select(mel_mask.unsqueeze(2))

        mel_loss = self.l1_loss(mel, mel_target)

        # print(duration_predicted.size())
        # print(duration_predictor_target.size())
        # print(src_mask.size())
        duration_predicted = duration_predicted.masked_select(src_mask)
        duration_predictor_target = duration_predictor_target.masked_select(
            src_mask)

        duration_predictor_loss = self.mse_loss(
            torch.log(1 + duration_predicted.squeeze()),
            torch.log(1 + duration_predictor_target.squeeze().float()))

        energy_predicted = energy_predicted.masked_select(
            mel_mask)
        energy_predictor_target = energy_predictor_target.masked_select(
            mel_mask)

        energy_predictor_loss = self.mse_loss(
            energy_predicted.squeeze(),
            energy_predictor_target.squeeze().float())

        pitch_predicted = pitch_predicted.masked_select(mel_mask)
        pitch_predictor_target = pitch_predictor_target.masked_select(
            mel_mask)

        pitch_predictor_loss = self.mse_loss(
            pitch_predicted.squeeze(),
            pitch_predictor_target.squeeze().float())

        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss
