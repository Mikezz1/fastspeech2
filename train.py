from fastspeech2.utils.utils import *
from fastspeech2.trainer.trainer import *
from fastspeech2.model.fastspeech import *
from fastspeech2.loss.loss import *
from fastspeech2.datasets.lj_speech import *
from fastspeech2.collate_fn.collate_fn import *
from fastspeech2.utils.utils import *
from fastspeech2.logger.wandb_writer import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from dataclasses import dataclass
from configs.base_config import *

# Init configs
mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()


# Load data
buffer = get_data_to_buffer(train_config)
collator = collate_fn_tensor(train_config)
dataset = BufferDataset(buffer)
training_loader = DataLoader(
    dataset,
    batch_size=train_config.batch_expand_size * train_config.batch_size,
    shuffle=True,
    collate_fn=collator,
    drop_last=True,
    num_workers=0
)
# add validation


# Load model
model = FastSpeech(model_config, mel_config, train_config)
model = model.to(train_config.device)


# Specify other stuff
fastspeech_loss = FastSpeechLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config.learning_rate,
    betas=(0.9, 0.98),
    eps=1e-9)

scheduler = OneCycleLR(optimizer, **{
    "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
    "epochs": train_config.epochs,
    "anneal_strategy": "cos",
    "max_lr": train_config.learning_rate,
    "pct_start": 0.1
})

logger = WanDBWriter(train_config)
trainer = Trainer(
    training_loader=training_loader, train_config=train_config, model=model,
    logger=logger, scheduler=scheduler, optimizer=optimizer,
    fastspeech_loss=fastspeech_loss)
trainer.train()
