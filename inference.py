import os
# os.chdir('../')


from fastspeech2.utils.utils import *
from fastspeech2.trainer.trainer import *
from fastspeech2.model.fastspeech import *
from fastspeech2.loss.loss import *
from fastspeech2.datasets.lj_speech import *
from fastspeech2.collate_fn.collate_fn import *
from fastspeech2.logger.wandb_writer import *
import torch
from configs.base_config import *
from fastspeech2.utils.text import text_to_sequence
import argparse
from vocoder import utils
from vocoder import waveglow


def get_data(train_config, n=-1):
    tests = [
        'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
        'Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition',
        'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
        'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space', '']
    data_list = list(text_to_sequence(test, train_config.text_cleaners)
                     for test in tests)

    return data_list[:n]


def synthesis(model, text, train_config, alpha=1.0, energy=1.0, pitch=1.0, ):
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    model.eval()
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha,
                            e_param=energy, p_param=pitch)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def main():
    args = parse_args()
    speed, energy, pitch = args.s, args.e, args.p

    mel_config = MelSpectrogramConfig()
    model_config = FastSpeechConfig()
    train_config = TrainConfig()

    model = FastSpeech(model_config, mel_config, train_config)
    model = model.to(train_config.device)
    print(os.getcwd())
    os.chdir('./vocoder')
    print(os.getcwd())

    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cpu()
    model.load_state_dict(
        torch.load(
            '../checkpoints/checkpoint_new_last_1.pth.tar', map_location='cpu')
        ['model'])
    model = model.eval()

    data_list = get_data(train_config, args.n)
    os.makedirs("results", exist_ok=True)
    for i, phn in tqdm(enumerate(data_list)):
        mel, mel_cuda = synthesis(
            model, phn, train_config, speed, energy, pitch, )

        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            f"results/test_final_s={speed}_p={pitch}_e={energy}_{i}_waveglow.wav"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=-1, type=int,
                        help="num of examples"
                        )
    parser.add_argument("-p", default=1, type=int,
                        help="pitch"
                        )
    parser.add_argument("-e", default=1, type=int,
                        help="energy"
                        )
    parser.add_argument("-s", default=1, type=int,
                        help="speed"
                        )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
