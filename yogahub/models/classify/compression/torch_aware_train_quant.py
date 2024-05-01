import torch
from neural_compressor import QuantizationAwareTrainingConfig
from neural_compressor.training import prepare_compression
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from yogahub.cfg.train import classify_config as config
from yogahub.data.classify import AlbumentationsDataset, extract_data, train_transform
from yogahub.models import YogaClassifier as YogaModel
from yogahub.utils import LabelSmoothing
from yogahub.utils.warmup import ExponentialWarmup

config.num_epochs = 5

train_img, train_label = extract_data(config.train_path)
train_dataset = AlbumentationsDataset(train_img, train_label, transform=train_transform)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
)


model = YogaModel(device=config.device, pretrained=config.pretrained)
conf = QuantizationAwareTrainingConfig()
compression_manager = prepare_compression(model.model.to("cpu"), conf)
compression_manager.callbacks.on_train_begin()
model = compression_manager.model

model.to(config.device)


# Model, Optimizer, and Loss
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.lr,
    betas=(0.9, 0.999),
    weight_decay=config.weight_decay,
)

# Warmup
num_steps = len(train_dataloader) * config.num_epochs
warmup_period = int(config.warmup_period * num_steps)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=warmup_period)

criterion_6 = LabelSmoothing(classes=6)
criterion_20 = LabelSmoothing(classes=20)
criterion_82 = LabelSmoothing(classes=82)


def train_epoch(epoch):
    model.train()
    for images, labels in tqdm(train_dataloader):
        lr_scheduler.step(lr_scheduler.last_epoch + 1)
        warmup_scheduler.step()

        l_6, l_20, l_82 = labels
        images, l_6, l_20, l_82 = (
            images.to(config.device),
            l_6.to(config.device),
            l_20.to(config.device),
            l_82.to(config.device),
        )

        # Forward pass
        p_82 = model(images)

        # Compute loss
        loss = criterion_82(p_82, l_82)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return None


for epoch in range(config.num_epochs):
    train_loss = train_epoch(epoch)


model.to("cpu")
model.eval()

compression_manager.callbacks.on_train_end()
compression_manager.save("./torch_quant")
