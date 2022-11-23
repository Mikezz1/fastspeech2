# imports
import os
from tqdm import tqdm
import numpy as np
from configs.base_config import *


def get_stats(config):
    pitch_vecs = []
    for pitch_file in tqdm(
            os.listdir(config.pitch_ground_truth),
            desc='computing stats for pitch'):
        file = np.load(os.path.join(config.pitch_ground_truth, pitch_file))
        pitch_vecs.append(file)

    pitch_vecs = np.concatenate(pitch_vecs)
    min, max, mean, std = pitch_vecs.min(), pitch_vecs.max(
    ), pitch_vecs.mean(), pitch_vecs.std()
    print(
        f"pitch: min: {min :.3f}, max: {max:.3f}, mean: {mean :.3f}, std:  {std :.3f}")

    energy_vecs = []
    for energy_file in tqdm(
            os.listdir(config.energy_ground_truth),
            desc='computing stats for energy'):
        file = np.load(os.path.join(config.energy_ground_truth, energy_file))
        energy_vecs.append(file)

    energy_vecs = np.concatenate(energy_vecs)
    min, max, mean, std = energy_vecs.min(), energy_vecs.max(
    ), energy_vecs.mean(), energy_vecs.std()
    print(
        f"energy: min: {min :.3f}, max: {max:.3f}, mean: {mean :.3f}, std:  {std :.3f}")

    alignment_vecs = []
    for alignment_file in tqdm(os.listdir(config.alignment_path),
                               desc='computing stats for alignments'):
        file = np.load(os.path.join(config.alignment_path, alignment_file))
        alignment_vecs.append(file)

    alignment_vecs = np.concatenate(alignment_vecs)
    min, max, mean, std = alignment_vecs.min(), alignment_vecs.max(
    ), alignment_vecs.mean(), alignment_vecs.std()
    print(
        f"alignment: min: {min :.3f}, max: {max:.3f}, mean: {mean :.3f}, std:  {std :.3f}")
    print(f"alignment == zero phonems frac: {np.mean(alignment_vecs == 0)}")


if __name__ == '__main__':
    config = TrainConfig()
    get_stats(config)
