# Extract data
import argparse
import os
import warnings

warnings.filterwarnings("ignore", message="Failed to load image Python extension")

import torch
from loguru import logger
from onnxruntime.quantization import CalibrationDataReader, quantize_static
from onnxruntime.quantization.preprocess import quant_pre_process
from torch.utils.data import DataLoader

from yogahub.cfg.train import classify_config as config
from yogahub.data.classify import AlbumentationsDataset, extract_data, test_transform

# reference: https://medium.com/@hdpoorna/pytorch-to-quantized-onnx-model-18cf2384ec27


# Get the metadata
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cuda:0", help="Running device")
    parser.add_argument(
        "-p",
        "--pretrained",
        default="yoga_model/convnext_small.in12k_ft_in1k_384.onnx",
        help="Pretrained ONNX path",
    )
    args = parser.parse_args()
    assert args.pretrained.endswith(".onnx"), "Invalid ONNX file path."
    return args


class QuantizationDataReader(CalibrationDataReader):
    def __init__(self, dataset, loader):
        self.test_dataset = dataset
        self.torch_dl = loader
        self.datasize = len(self.torch_dl)
        self.enum_data = iter(self.torch_dl)

    def get_next(self):
        batch = next(self.enum_data, None)
        return {"input": batch.cpu().numpy()} if batch is not None else None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)


def main():
    args = parse_arguments()

    if not torch.cuda.is_available() and "cuda" in args.device:
        args.device = "cpu"

    test_img, test_label = extract_data(config.test_path)

    test_dataset = AlbumentationsDataset(
        test_img, test_label, transform=test_transform, return_label=False
    )

    loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    qdr = QuantizationDataReader(test_dataset, loader)

    activation_symmetric = True if "cuda" in args.device else False
    q_static_opts = {
        "ActivationSymmetric": activation_symmetric,
        "WeightSymmetric": True,
    }
    model_int8_path = args.pretrained.replace(".onnx", "_quant.onnx")
    model_preprocess = args.pretrained.replace(".onnx", "_pre.onnx")

    logger.info(f"Preprocess Model")
    quant_pre_process(args.pretrained, model_preprocess)

    logger.info(f"Start Quantization: Model: {args.pretrained}, Device: {args.device}")
    quantized_model = quantize_static(
        model_input=model_preprocess,
        model_output=model_int8_path,
        calibration_data_reader=qdr,
        extra_options=q_static_opts,
    )

    os.remove(model_preprocess)
    logger.info("Finished Quantization")
    logger.info(f"Save Quantization Model: {model_int8_path}")


if __name__ == "__main__":
    main()
