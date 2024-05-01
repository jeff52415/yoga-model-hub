import torch
from loguru import logger
from neural_compressor import quantization
from neural_compressor.config import (
    AccuracyCriterion,
    PostTrainingQuantConfig,
    TuningCriterion,
)
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from yogahub.cfg.train import classify_config as config
from yogahub.data.classify import AlbumentationsDataset, extract_data, test_transform
from yogahub.models import YogaClassifier as YogaModel

# Quantization code

config.device = "cpu"
test_img, test_label = extract_data(config.test_path)
test_img = test_img[: config.batch_size * 15]
test_label = test_label[: config.batch_size * 15]

test_dataset = AlbumentationsDataset(
    test_img, test_label, transform=test_transform, return_label=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
)


calibration_dataset = AlbumentationsDataset(
    test_img, test_label, transform=test_transform, return_label=False
)

calibration_loader = DataLoader(
    calibration_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=config.pin_memory,
)

yogamodel = YogaModel(device=config.device, pretrained=config.pretrained)
model = yogamodel.model


def evaluate(model, test_loader=test_loader):
    model.eval()
    all_true_82, all_pred_82 = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            l_6, l_20, l_82 = labels
            images, l_6, l_20, l_82 = (
                images.to(config.device),
                l_6.to(config.device),
                l_20.to(config.device),
                l_82.to(config.device),
            )

            # Forward pass
            p_6, p_20, p_82 = model(images)
            _, preds_82 = torch.max(torch.softmax(p_82, dim=1), 1)

            all_true_82.extend(l_82.cpu().numpy())
            all_pred_82.extend(preds_82.cpu().numpy())

    # Compute metrics
    accuracy_82 = accuracy_score(all_true_82, all_pred_82)

    return accuracy_82


# reference: https://github.com/intel/neural-compressor/blob/master/docs/source/quantization.md
# PyTorch : https://pytorch.org/docs/stable/quantization.html#prototype-fx-graph-mode-quantization
logger.info(f"Start Qauntization: {config.pretrained}")

recipes = {
    "smooth_quant": False,
    "smooth_quant_args": {
        "alpha": 0.5,
    },  # default value is 0.5
    "fast_bias_correction": False,
}
accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)


conf = PostTrainingQuantConfig(
    approach="static",
    calibration_sampling_size=[config.batch_size],
    tuning_criterion=TuningCriterion(timeout=0, max_trials=100),
    accuracy_criterion=accuracy_criterion,
    recipes=recipes,
)  # default approach is "auto", you can set "dynamic":PostTrainingQuantConfig(approach="dynamic")


q_model = quantization.fit(
    model=model,
    conf=conf,
    eval_func=evaluate,
    calib_dataloader=calibration_loader,
)


q_model.save("./neural_compressor")
logger.info(f"Complete Qauntization: store model in neural_compressor")
