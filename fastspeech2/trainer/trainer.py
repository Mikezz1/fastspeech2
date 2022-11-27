import PIL
from torchvision.transforms import ToTensor
import random
import torch
from torch import nn
import os
from tqdm import tqdm
from fastspeech2.loss.loss import *
from fastspeech2.logger.utils import *
from fastspeech2.utils.text import text_to_sequence, sequence_to_text
import numpy as np
from g2p_en import G2p


class Trainer:

    def __init__(
            self, training_loader, train_config, model, logger, optimizer,
            scheduler, fastspeech_loss):
        self.training_loader = training_loader
        self.train_config = train_config
        self.model = model
        self.logger = logger
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fastspeech_loss = fastspeech_loss

    def _log_spectrogram(self, spectrogram, caption='spectrogram_t'):
        spectrogram = spectrogram.detach().cpu()
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.logger.add_image(caption, ToTensor()(image))

    def _log_audio(self, audio, caption='audio_t'):
        self.logger.add_audio(
            caption, audio.detach().cpu(),
            sample_rate=22_050)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    @torch.no_grad()
    def inference(
            self, model, text, train_config, g2p, alpha=1, energy=1, pitch=1):
        model.eval()
        with torch.no_grad():
            text = ' '.join(g2p(text))
            char = text_to_sequence(text, train_config.text_cleaners)
            text = np.array(char)
            text = np.stack([text])
            src_pos = np.array([i+1 for i in range(text.shape[1])])
            src_pos = np.stack([src_pos])
            sequence = torch.from_numpy(text).long().to(train_config.device)
            src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
            mel = model.forward(
                sequence,
                src_pos,
                alpha=alpha,
                e_param=energy,
                p_param=pitch)

            model.train()
        return mel

    def train(self):
        current_step = 0
        tqdm_bar = tqdm(total=self.train_config.epochs * len(self.training_loader)
                        * self.train_config.batch_expand_size - current_step)
        g2p = G2p()

        for epoch in range(self.train_config.epochs):
            for i, batchs in enumerate(self.training_loader):
                # real batch start here
                for j, db in enumerate(batchs):
                    current_step += 1
                    tqdm_bar.update(1)

                    self.logger.set_step(current_step)

                    # Get Data
                    character = db["text"].long().to(self.train_config.device)
                    mel_target = db["mel_target"].float().to(
                        self.train_config.device)
                    duration = db["duration"].int().to(
                        self.train_config.device)
                    mel_pos = db["mel_pos"].long().to(self.train_config.device)
                    src_pos = db["src_pos"].long().to(self.train_config.device)
                    max_mel_len = db["mel_max_len"]

                    energy_target = db["energy_target"].float().to(
                        self.train_config.device)

                    pitch_target = db["pitch_target"].float().to(
                        self.train_config.device)

                    # Forward
                    mel_output, duration_predictor_output,\
                        energy_predictor_output, pitch_predictor_output, mel_mask, src_mask = self.model(
                            character, src_pos, mel_pos=mel_pos,
                            mel_max_length=max_mel_len, length_target=duration, energy_target=energy_target, pitch_target=pitch_target)

                    # Calc Loss
                    mel_loss, duration_loss, energy_loss, pitch_loss = self.fastspeech_loss(
                        mel_output, duration_predictor_output, energy_predictor_output, pitch_predictor_output,
                        mel_target, duration, energy_target, pitch_target, mel_mask, src_mask)
                    total_loss = mel_loss + duration_loss + energy_loss + pitch_loss

                    # Backward
                    total_loss.backward()

                    # Logger

                    if current_step % self.train_config.log_step == 0:
                        t_l = total_loss.detach().cpu().numpy()
                        m_l = mel_loss.detach().cpu().numpy()
                        d_l = duration_loss.detach().cpu().numpy()
                        e_l = energy_loss.detach().cpu().numpy()
                        p_l = pitch_loss.detach().cpu().numpy()

                        self.logger.add_scalar("duration_loss", d_l)
                        self.logger.add_scalar("mel_loss", m_l)
                        self.logger.add_scalar("energy_loss", e_l)
                        self.logger.add_scalar("pitch_loss", p_l)
                        self.logger.add_scalar("total_loss", t_l)
                        self.logger.add_scalar(
                            "learning_rate", self.scheduler.get_lr()[0])
                        self.logger.add_scalar(
                            "grad_norm", self.get_grad_norm())
                        self.logger.add_scalar("epoch", epoch)
                        self._log_spectrogram(
                            mel_output[0][mel_mask[0],
                                          :].T,
                            caption='predicted spectrogram')
                        self._log_spectrogram(
                            mel_target[0][mel_mask[0], :].T, caption='gt spectrogram')
                        self._log_audio(
                            get_inv_mel_spec(mel_output[0][mel_mask[0], :]),
                            caption='predicted audio (istft)')
                        self._log_audio(
                            get_inv_mel_spec(mel_target[0][mel_mask[0], :]),
                            caption='gt audio (istft)')

                        # if current_step > 20:

                        sample_text = 'defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest'

                        sample_mel = self.inference(
                            self.model, sample_text, self.train_config, g2p)

                        self._log_audio(
                            get_inv_mel_spec(sample_mel[0]),
                            caption='sample audio (istft)')
                        self._log_spectrogram(
                            sample_mel[0].T, caption='sample spectrogram')

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_config.grad_clip_thresh)

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()

                    if current_step % self.train_config.save_step == 0:
                        torch.save(
                            {'model': self.model.state_dict(),
                             'optimizer': self.optimizer.state_dict()},
                            os.path.join(
                                self.train_config.checkpoint_path,
                                f'checkpoint_new_last_{(current_step + 1) % 10}.pth.tar'))
                        print("save model at step %d ..." % current_step)
