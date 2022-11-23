import torch
import torch.nn as nn
from fastspeech2.model.blocks import *


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config, device):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VarianceAdaptor(model_config)
        self.device = device

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_prediction_output = self.duration_predictor(x)
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_prediction_output
        else:
            duration_prediction_output = (
                duration_prediction_output * alpha + 0.5).int()

            output = self.LR(x, duration_prediction_output)
            mel_pos = torch.stack([torch.Tensor(
                [i+1 for i in range(output.size(1))])]).long().to(self.device)
            return output, mel_pos


class EnergyAdaptor(nn.Module):
    """ Energy adaptor
    Add quantization
    """

    def __init__(self,  model_config, train_config, device):
        super(EnergyAdaptor, self).__init__()
        self.energy_predictor = VarianceAdaptor(model_config)
        self.device = device
        self.energy_embedding = nn.Embedding(256, model_config.encoder_dim)
        bin_min = (
            train_config.energy_min - train_config.energy_mean) / train_config.energy_std
        bin_max = (
            train_config.energy_max - train_config.energy_mean) / train_config.energy_std
        self.energy_bins = nn.Parameter(
            torch.exp(
                torch.linspace(bin_min, bin_max, 256 - 1)
            ),
            requires_grad=False,
        )

    def forward(self, x, target=None):
        energy_predictions = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embedding(
                torch.bucketize(target, self.energy_bins))
            #x = x+embedding
            return embedding, energy_predictions
        else:
            embedding = self.energy_embedding(
                torch.bucketize(energy_predictions, self.energy_bins))
            #x += embedding
            return embedding


class PitchAdaptor(nn.Module):
    """ Energy adaptor
    Add quantization
    """

    def __init__(self,  model_config, train_config, device):
        super(PitchAdaptor, self).__init__()
        self.pitch_predictor = VarianceAdaptor(model_config)
        self.device = device
        self.pitch_embedding = nn.Embedding(256, model_config.encoder_dim)
        bin_min = (
            train_config.pitch_min - train_config.pitch_mean) / train_config.pitch_std
        bin_max = (
            train_config.pitch_max - train_config.pitch_mean) / train_config.pitch_std
        self.pitch_bins = nn.Parameter(
            torch.exp(
                torch.linspace(bin_min, bin_max, 256 - 1)
            ),
            requires_grad=False,
        )

    def forward(self, x, target=None):
        pitch_predictions = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_embedding(
                torch.bucketize(target, self.pitch_bins))
            #x = x+embedding
            return embedding, pitch_predictions
        else:
            embedding = self.pitch_embedding(
                torch.bucketize(pitch_predictions, self.pitch_bins))
            #x += embedding
            return embedding


class VarianceAdaptor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat
