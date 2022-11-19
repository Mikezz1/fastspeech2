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
    mel_ground_truth = "./data/mels"
    alignment_path = "./data/alignments"
    data_path = './data/train.txt'

    wandb_project = 'fastspeech_example'

    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 1
    epoch_len = 1
    epochs = 500
    n_warm_up_step = 30

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 10.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 100
    log_step = 20
    clear_Time = 20

    batch_expand_size = 1
