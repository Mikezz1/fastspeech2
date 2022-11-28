# imports
import os
from tqdm import tqdm
import torchaudio
from vocoder.audio.stft import TacotronSTFT
import torch
import numpy as np
import pyworld as pw
from configs.base_config import *
# main


def calc_pitch(audio, config):
    audio = audio.squeeze().numpy().astype(np.float64)
    pitch, _ = pw.dio(
        audio,
        config.sampling_rate,
        frame_period=config.hop_length / config.sampling_rate * 1000,
    )
    return pitch


def main(config):
    # read wav
    os.makedirs(config.mel_ground_truth, exist_ok=True)
    os.makedirs(config.pitch_ground_truth, exist_ok=True)
    os.makedirs(config.energy_ground_truth, exist_ok=True)

    wavs_dir = sorted(os.listdir(config.audio_ground_truth))
    for audio_path in tqdm(wavs_dir, desc='process wavs'):
        waveform, _ = torchaudio.load(os.path.join(
            config.audio_ground_truth,
            audio_path),
            normalize=True)

        # compute spectrogram
        melspec_transform = TacotronSTFT(
            filter_length=config.filter_length, hop_length=config.hop_length,
            win_length=config.win_length, n_mel_channels=config.n_mel_channels,
            sampling_rate=config.sampling_rate)

        mel, magnitude = melspec_transform.mel_spectrogram(waveform)
        mel, magnitude = mel.squeeze(0).numpy(), magnitude.squeeze(0)
        energy = torch.linalg.norm(magnitude, dim=0,  ord=2).numpy()
        pitch = calc_pitch(waveform, config)

        # save everything
        np.save(
            f"{config.mel_ground_truth}/{audio_path.split('.')[0]}_mel.npy",
            mel, allow_pickle=False)

        np.save(
            f"{config.energy_ground_truth}/{audio_path.split('.')[0]}_e.npy",
            energy, allow_pickle=False)
        np.save(
            f"{config.pitch_ground_truth}/{audio_path.split('.')[0]}_p.npy",
            pitch, allow_pickle=False)


if __name__ == '__main__':
    config = TrainConfig()
    main(config)
