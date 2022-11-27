import torch
import numpy as np
from fastspeech2.utils.text import text_to_sequence
import time
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from fastspeech2.utils.utils import *
from fastspeech2.utils.text.symbols import symbols


def calc_pitch(audio):
    raise NotImplementedError


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config.data_path,
                        num_objects=train_config.epoch_len)

    wavs_dir = sorted(os.listdir(train_config.audio_ground_truth))
    start = time.perf_counter()
    for i, audio_path in tqdm(zip(range(len(text)), wavs_dir)):

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth,
            f"{audio_path.split('.')[0]}_mel.npy")
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))

        if train_config.use_mfa:
            character = text[i][:-1]
            _symbol_to_id = {s: i for i, s in enumerate(symbols)}
            # print(_symbol_to_id)
            seq = []
            for symbol in character.split(' '):
                if symbol == "spn":
                    symbol = " "
                    seq.append(_symbol_to_id[symbol])
                else:
                    seq.append(_symbol_to_id['@' + symbol])

            character = np.array(seq)

        else:
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, train_config.text_cleaners))

        assert len(character) == len(duration)

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target).T

        # add pitch and energy targets
        energy_target = np.load(os.path.join(
            train_config.energy_ground_truth,
            f"{audio_path.split('.')[0]}_e.npy"))
        pitch_target = np.load(os.path.join(
            train_config.pitch_ground_truth,
            f"{audio_path.split('.')[0]}_p.npy"))

        energy_target = torch.from_numpy(energy_target)
        pitch_target = torch.from_numpy(pitch_target)

        # transform targets
        energy_target = (
            energy_target - train_config.energy_mean) / train_config.energy_std
        pitch_target = (
            pitch_target - train_config.pitch_mean) / train_config.pitch_std

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target,
                       "energy_target": energy_target,
                       "pitch_target": pitch_target})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    energy_targets = [batch[ind]["energy_target"] for ind in cut_list]
    pitch_targets = [batch[ind]["pitch_target"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(
            np.pad(
                [i + 1 for i in range(int(length_mel_row))],
                (0, max_mel_len - int(length_mel_row)),
                'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    energy_targets = pad_1D_tensor(energy_targets)
    pitch_targets = pad_1D_tensor(pitch_targets)
    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
           "energy_target": energy_targets,
           "pitch_target": pitch_targets}

    return out
