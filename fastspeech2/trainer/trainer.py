import PIL
from torchvision.transforms import ToTensor
import random
import torch
from torch import nn
import os
from tqdm import tqdm
from fastspeech2.loss.loss import *
from fastspeech2.logger.utils import *


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

    def train(self):
        current_step = 0
        tqdm_bar = tqdm(total=self.train_config.epochs * len(self.training_loader)
                        * self.train_config.batch_expand_size - current_step)
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

                    # Forward
                    mel_output, duration_predictor_output = self.model(
                        character, src_pos, mel_pos=mel_pos,
                        mel_max_length=max_mel_len, length_target=duration)

                    # Calc Loss
                    mel_loss, duration_loss = self.fastspeech_loss(
                        mel_output, duration_predictor_output, mel_target, duration)
                    total_loss = mel_loss + duration_loss

                    # Backward
                    total_loss.backward()

                    # Logger
                    t_l = total_loss.detach().cpu().numpy()
                    m_l = mel_loss.detach().cpu().numpy()
                    d_l = duration_loss.detach().cpu().numpy()

                    self.logger.add_scalar("duration_loss", d_l)
                    self.logger.add_scalar("mel_loss", m_l)
                    self.logger.add_scalar("total_loss", t_l)
                    self.logger.add_scalar("grad_norm", self.get_grad_norm())
                    self.logger.add_scalar("epoch", epoch)
                    self._log_spectrogram(
                        mel_output[0].T, caption='predicted spectrogram')
                    self._log_spectrogram(
                        mel_target[0].T, caption='gt spectrogram')
                    self._log_audio(
                        get_inv_mel_spec(mel_output[0]),
                        caption='predicted audio (istft)')
                    self._log_audio(
                        get_inv_mel_spec(mel_target[0]),
                        caption='gt audio (istft)')

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
                                'checkpoint_%d.pth.tar' % current_step))
                        print("save model at step %d ..." % current_step)
