
import numpy as np
import tgt
import os
# import pandas as pd
from tqdm import tqdm


class Aligner:
    def __init__(self, hop_length, sampling_rate):
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.sil_phones = ["sil", "sp", "spn"]

    def __call__(self, grid):
        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in grid._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in self.sil_phones:
                    continue
                else:
                    start_time = s

            if p not in self.sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        phones = ' '.join(phones)

        return phones, durations


if __name__ == '__main__':
    phones_train = []

    save_dir = 'data'
    os.makedirs('data/mfa_alignments', exist_ok=True)

    for i, sample in tqdm(
            enumerate(sorted(os.listdir('data/ljs_aligned')))):
        textgrid = tgt.io.read_textgrid(
            f'data/ljs_aligned/{sample}')
        aligner = Aligner(hop_length=256, sampling_rate=22_050)

        phones, durations = aligner(
            textgrid.get_tier_by_name('phones'))
        phones_train.append(phones)

        durations = np.array(durations)

        # print(len(durations))
        # print(len(phones.split(' ')))
        assert len(durations) == len(phones.split(' '))

        np.save(
            f"{save_dir}/mfa_alignments/{i}.npy",
            durations, allow_pickle=False)

    with open(f"{save_dir}/train_phones.txt", 'w') as f:
        for sample in phones_train:
            f.write(f"{sample}\n")

    print(f'Phones: {phones}')
    print(f'Durations: {durations}')
