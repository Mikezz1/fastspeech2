import torch
from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    num_mels = 80


@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 128
    encoder_n_layer = 2
    encoder_head = 2
    encoder_conv1d_filter_size = 512

    decoder_dim = 128
    decoder_n_layer = 2
    decoder_head = 2
    decoder_conv1d_filter_size = 512

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


@dataclass
class TrainConfig:
    checkpoint_path = "./checkpoints"
    logger_path = "./logger"

    mel_ground_truth = "./data/melspecs"
    pitch_ground_truth = "./data/pitch"
    energy_ground_truth = "./data/energy"

    audio_ground_truth = "./data/LJSpeech-1.1/wavs"
    alignment_path = "./data/alignments"
    data_path = './data/train.txt'

    wandb_project = 'fastspeech_example'

    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 4
    epoch_len = 4
    epochs = 25
    n_warm_up_step = 3000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 1
    log_step = 5
    clear_Time = 20

    batch_expand_size = 1

    hop_length = 256
    win_length = 1024
    filter_length = 1024
    sampling_rate = 22_500
    n_mel_channels = 80

    energy_mean = 31.755
    energy_min = 0.115
    energy_max = 227.832
    energy_std = 29.441

    pitch_mean = 149.885
    pitch_min = 0.000
    pitch_max = 674.820
    pitch_std = 125.962
