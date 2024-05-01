import os
import warnings
from collections import Counter

import torch
import torch.distributed as dist
from loguru import logger
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from yogahub.cfg.train import classify_config as config
from yogahub.data.classify import (
    AlbumentationsDataset,
    extract_data,
    test_transform,
    train_transform,
)
from yogahub.models.classify.backbone import TimmModelWrapper, dinov2_vitb14_lc
from yogahub.utils import LabelSmoothing
from yogahub.utils.utils import load_model_weights
from yogahub.utils.warmup import ExponentialWarmup

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = config.device
os.environ["OMP_NUM_THREADS"] = "1"
# nohup torchrun --nproc_per_node=3 train.py &
# tmux new -d -s yoga 'torchrun --nproc_per_node=3 train.py'
# tmux pipe-pane -o 'cat >>~/tmux.log'
# tmux ls

# Initialize the distributed environment
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="run")
pretrained = config.pretrained

if config.model in ["dino2_vit_base", "dino2_vit_small"]:
    model = dinov2_vitb14_lc(
        multiclassifier=[6, 20, 82],
        version=config.model,
        drop_path_rate=config.drop_path_rate,
    )

elif config.model in [
    "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
    "convnext_small.in12k_ft_in1k_384",
]:
    model = TimmModelWrapper(
        timm_model=config.model,
        pretrained=not pretrained,
        drop_path_rate=config.drop_path_rate,
    )

if pretrained:
    model = load_model_weights(model, pretrained, strict=False)
    logger.info(f"Load pretrained weight from: {pretrained}")

model.to(device)
model = DistributedDataParallel(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True,
)
# output = model.backbone.get_intermediate_layers(torch.randn(1, 3, 518, 518).to(device), n=4, return_class_token=True)
# output = model.forward(torch.randn(1, 3, 518, 518).to(device))


# Extract data
train_img, train_label = extract_data(config.train_path)
test_img, test_label = extract_data(config.test_path)

# Separate labels for different classes
c_6, c_20, c_82 = zip(*train_label)

# Compute class counts for c_82
class_count_dict = Counter(c_82)
class_count = [class_count_dict[i] for i in range(len(class_count_dict))]

# Datasets and Dataloaders
train_dataset = AlbumentationsDataset(train_img, train_label, transform=train_transform)
class_weights = 1.0 / torch.tensor(class_count, dtype=torch.float)
sample_weights = class_weights[list(c_82)]
sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)


train_sampler = DistributedSampler(train_dataset)

train_dataloader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
)


test_dataset = AlbumentationsDataset(test_img, test_label, transform=test_transform)
test_sampler = DistributedSampler(test_dataset)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    sampler=test_sampler,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
)

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
    total_loss = 0
    for images, labels in tqdm(train_dataloader):
        lr_scheduler.step(lr_scheduler.last_epoch + 1)
        warmup_scheduler.step()

        l_6, l_20, l_82 = labels
        images, l_6, l_20, l_82 = (
            images.to(device),
            l_6.to(device),
            l_20.to(device),
            l_82.to(device),
        )

        # Forward pass
        p_6, p_20, p_82 = model(images)

        # Compute loss
        loss_6 = criterion_6(p_6, l_6)
        loss_20 = criterion_20(p_20, l_20)
        loss_82 = criterion_82(p_82, l_82)
        loss = loss_6 + loss_20 + loss_82

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(epoch):
    model.eval()
    total_loss = 0
    all_true_6, all_pred_6 = [], []
    all_true_20, all_pred_20 = [], []
    all_true_82, all_pred_82 = [], []

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader):
            l_6, l_20, l_82 = labels
            images, l_6, l_20, l_82 = (
                images.to(device),
                l_6.to(device),
                l_20.to(device),
                l_82.to(device),
            )

            # Forward pass
            p_6, p_20, p_82 = model(images)

            # Compute loss
            loss_6 = criterion_6(p_6, l_6)
            loss_20 = criterion_20(p_20, l_20)
            loss_82 = criterion_82(p_82, l_82)
            loss = loss_6 + loss_20 + loss_82

            total_loss += loss.item()

            # Calculate metrics
            _, preds_6 = torch.max(torch.softmax(p_6, dim=1), 1)
            _, preds_20 = torch.max(torch.softmax(p_20, dim=1), 1)
            _, preds_82 = torch.max(torch.softmax(p_82, dim=1), 1)

            all_true_6.extend(l_6.cpu().numpy())
            all_pred_6.extend(preds_6.cpu().numpy())
            all_true_20.extend(l_20.cpu().numpy())
            all_pred_20.extend(preds_20.cpu().numpy())
            all_true_82.extend(l_82.cpu().numpy())
            all_pred_82.extend(preds_82.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader)

    # Compute metrics
    accuracy_6 = accuracy_score(all_true_6, all_pred_6)
    precision_6 = precision_score(all_true_6, all_pred_6, average="macro")
    recall_6 = recall_score(all_true_6, all_pred_6, average="macro")

    accuracy_20 = accuracy_score(all_true_20, all_pred_20)
    precision_20 = precision_score(all_true_20, all_pred_20, average="macro")
    recall_20 = recall_score(all_true_20, all_pred_20, average="macro")

    accuracy_82 = accuracy_score(all_true_82, all_pred_82)
    precision_82 = precision_score(all_true_82, all_pred_82, average="macro")
    recall_82 = recall_score(all_true_82, all_pred_82, average="macro")

    writer.add_scalar("Loss/eval", avg_loss, epoch)
    writer.add_scalar("Accuracy/eval_6", accuracy_6, epoch)
    writer.add_scalar("Accuracy/eval_20", accuracy_20, epoch)
    writer.add_scalar("Accuracy/eval_82", accuracy_82, epoch)
    writer.add_scalar("Precision/eval_6", precision_6, epoch)
    writer.add_scalar("Precision/eval_20", precision_20, epoch)
    writer.add_scalar("Precision/eval_82", precision_82, epoch)
    writer.add_scalar("Recall/eval_6", recall_6, epoch)
    writer.add_scalar("Recall/eval_20", recall_20, epoch)
    writer.add_scalar("Recall/eval_82", recall_82, epoch)

    logger.info(f"\n--- Evaluation after Epoch {epoch + 1} ---")
    logger.info(f"Eval Loss: {avg_loss:.4f}")
    logger.info(
        f"6-Classes  -> Accuracy: {accuracy_6:.4f}, Precision: {precision_6:.4f}, Recall: {recall_6:.4f}"
    )
    logger.info(
        f"20-Classes -> Accuracy: {accuracy_20:.4f}, Precision: {precision_20:.4f}, Recall: {recall_20:.4f}"
    )
    logger.info(
        f"82-Classes -> Accuracy: {accuracy_82:.4f}, Precision: {precision_82:.4f}, Recall: {recall_82:.4f}\n"
    )

    return avg_loss, (
        (accuracy_6, precision_6, recall_6),
        (accuracy_20, precision_20, recall_20),
        (accuracy_82, precision_82, recall_82),
    )


# Training loop with epochs
if not os.path.exists(config.weight_path):
    os.makedirs(config.weight_path, exist_ok=True)

weight_path = os.path.join(config.weight_path, f"checkpoint_{config.model}.pth")

if pretrained:
    weights = torch.load(pretrained, map_location=torch.device("cpu"))
    best_accuracy = weights.get("accuracy_82")
    logger.info(f"start from accuracy: {best_accuracy}")
else:
    best_accuracy = 0

for epoch in range(config.num_epochs):
    train_loss = train_epoch(epoch)
    logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Train Loss: {train_loss:.4f}")

    test_loss, metrics = evaluate(epoch)
    class_6, class_20, class_82 = metrics
    accuracy_6, _, _ = class_6
    accuracy_20, _, _ = class_20
    accuracy_82, _, _ = class_82

    if accuracy_82 > best_accuracy:
        best_accuracy = accuracy_82
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": dict(config),
                "train_loss": round(train_loss, 4),
                "test_loss": round(test_loss, 4),
                "accuracy_82": round(accuracy_82, 4),
                "accuracy_20": round(accuracy_20, 4),
                "accuracy_6": round(accuracy_6, 4),
            },
            weight_path,
        )
        logger.info(f"Save checkpoint successfully! Accuracy_82: {accuracy_82:.4f}")
