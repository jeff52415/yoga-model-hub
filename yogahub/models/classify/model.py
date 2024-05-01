import os
from typing import List, Optional, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image

from yogahub import ROOT
from yogahub.data.classify.augmentation import test_transform
from yogahub.models.classify.backbone import TimmModelWrapper, dinov2_vitb14_lc
from yogahub.utils.utils import load_from_yaml, load_model_weights, read_poses_from_file

FIRST_LEVEL_INDEX_EN = ROOT / "metadata/first_level_index_en.yaml"
FIRST_LEVEL_EN_CH = ROOT / "metadata/first_level_en_ch.yaml"
POSTS_DATA = ROOT / "metadata/pose.txt"
THIRD_LEVEL_EN_CH = ROOT / "metadata/third_level_en_ch.yaml"
PRETRAINED = ROOT / "weight/classify/convnext_small.in12k_ft_in1k_384.pth"


class YogaClassifier:
    def __init__(
        self,
        backbone="convnext_small.in12k_ft_in1k_384",
        pretrained: str = PRETRAINED,
        device: str = "cuda:0",
    ):
        if not torch.cuda.is_available():
            device = "cpu"
        pretrained = str(pretrained)
        self.device = device
        self.pretrained = pretrained
        self.first_level_index_en = load_from_yaml(FIRST_LEVEL_INDEX_EN)
        self.first_level_en_ch = load_from_yaml(FIRST_LEVEL_EN_CH)
        self.third_level_en_ch = load_from_yaml(THIRD_LEVEL_EN_CH)
        self.poses_data = read_poses_from_file(POSTS_DATA)
        self.onnx = False

        if backbone in ["dino2_vit_base", "dino2_vit_small"]:
            self.model = dinov2_vitb14_lc(
                multiclassifier=[6, 20, 82], pretrained=False, version=backbone
            )
        elif backbone in [
            "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
            "convnext_small.in12k_ft_in1k_384",
        ]:
            self.model = TimmModelWrapper(
                timm_model=backbone,
                pretrained=False,
            )
        if pretrained:
            if pretrained.endswith(".pth"):
                self.model = load_model_weights(self.model, pretrained, strict=True)
                logger.info(f"Load pretrained weight from: {pretrained}")
            elif pretrained.endswith(".pt"):
                try:
                    self.model = torch.jit.load(pretrained)
                    logger.info(f"Loaded TorchScript Module from: {pretrained}")
                except:
                    from neural_compressor.utils.pytorch import load

                    int8_model = load(pretrained, self.model)
                    self.model = int8_model
                    logger.info(
                        f"Loaded Neural Compressor Quant Model from: {pretrained}"
                    )
            elif pretrained.endswith(".onnx"):
                import onnxruntime as ort

                # Load the ONNX model
                # onnx_model = onnx.load(pretrained)
                # onnx.checker.check_model(onnx_model)
                self.onnx = True
                if "cuda" in device:
                    device_number = device.split(":")[1]
                    self.model = ort.InferenceSession(
                        pretrained,
                        providers=["CUDAExecutionProvider"],
                        provider_options=[
                            {"device_id": device_number}
                        ],  # Specify the device ID of the CUDA device
                    )
                else:
                    self.model = ort.InferenceSession(
                        pretrained,
                        providers=["CPUExecutionProvider"],
                    )
                logger.info("Load ONNX Module")
                logger.info(f"Load pretrained weight from: {pretrained}")

            else:
                logger.info("No pretrained provid")

        if not self.onnx:
            self.model.to(device)
            self.model.eval()

    def tensor_to_numpy(self, tensor):
        """
        Converts a PyTorch tensor to a NumPy array.
        If the tensor requires gradient computation, it detaches the tensor first.

        Parameters:
        tensor (torch.Tensor): The tensor to convert.

        Returns:
        numpy.ndarray: The resulting NumPy array.
        """
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Input should be a PyTorch tensor.")

        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    def benchmark_model(
        self, input_data: Optional[torch.Tensor] = None, num_trials: int = 200
    ) -> float:
        """
        Benchmarks the model by running inference for a specified number of trials.

        Parameters:
        input_data (torch.Tensor): The input data for inference.
        num_trials (int, optional): The number of trials for the benchmark. Default is 200.

        Returns:
        float: The average time per inference trial.
        """
        import time

        if not input_data:
            input_data = torch.randn(4, 3, 448, 448)
        input_data = input_data.to(self.device)

        warm_up = num_trials // 5

        for _ in range(warm_up):
            if self.onnx:
                ort_inputs = {
                    self.model.get_inputs()[0].name: self.tensor_to_numpy(input_data)
                }
                # Run inference
                self.model.run(None, ort_inputs)
            else:
                with torch.no_grad():
                    self.model(input_data)

        start_time = time.perf_counter()
        for _ in range(num_trials):
            if self.onnx:
                ort_inputs = {
                    self.model.get_inputs()[0].name: self.tensor_to_numpy(input_data)
                }
                # Run inference
                self.model.run(None, ort_inputs)
            else:
                with torch.no_grad():
                    self.model(input_data)
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / num_trials

        # Log the required information
        logger.info("Benchmark Result")
        logger.info(f"Input Data: {input_data.shape}")
        logger.info(f"Num Trails: {num_trials}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Weight: {self.pretrained}")
        logger.info(f"Model Type: {type(self.model)}")
        logger.info(f"Time: {avg_time:.6f} seconds per inference")
        return avg_time

    def export(
        self,
        save_path: Optional[str] = "",
        convert_type: str = "torchscript",
        input_size: tuple = (1, 3, 448, 448),
    ) -> None:
        """
        Exports the model to TorchScript format and saves it to disk.

        Args:
            convert_type (str): Convert the model to TorchScript/Onnx. Default: torchscript.
            input_size (tuple): The size of the input tensor for tracing. Default: (1, 3, 448, 448).

        Returns:
            None
        """

        # torchscript: https://pytorch.org/docs/stable/jit.html
        if self.pretrained.endswith(".pt"):
            logger.info(
                f"Model file {self.pretrained} already has a .pt extension. Skipping export."
            )
            return None

        if convert_type == "torchscript":
            try:
                example_input = torch.randn(input_size).to(
                    self.device
                )  # Create a random tensor with the specified input size
                traced_script_module = torch.jit.script(
                    self.model, example_input
                )  # Script the model
                if not save_path:
                    save_path = self.pretrained.replace(
                        ".pth", ".pt"
                    )  # Replace the file extension
                traced_script_module.save(
                    save_path
                )  # Save the TorchScript module to disk
                logger.info(f"Successfully converted model to TorchScript: {save_path}")
                logger.info(traced_script_module.code)  # Log the TorchScript code
            except Exception as e:
                logger.error(f"Failed to export model to TorchScript: {e}")
        elif convert_type == "onnx":
            try:
                example_input = torch.randn(input_size, requires_grad=True).to(
                    self.device
                )  # Create a random tensor with the specified input size
                torch_out = self.model(example_input)
                output_names = [f"output_{i}" for i in range(len(torch_out))]
                input_names = [
                    "input"
                ]  # Assuming there's a single input. Adjust as necessary.
                if not save_path:
                    save_path = self.pretrained.replace(
                        ".pth", ".onnx"
                    )  # Replace the file extension
                torch.onnx.export(
                    model=self.model,  # model being run
                    args=example_input,  # model input (or a tuple for multiple inputs)
                    f=save_path,  # where to save the model (can be a file or file-like object)
                    export_params=True,  # store the trained parameter weights inside the model file
                    opset_version=18,  # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names=input_names,  # name of the input tensor(s)
                    output_names=output_names,  # name of the output tensor(s)
                    dynamic_axes={
                        "input": {0: "batch"},  # dynamic axes for input
                        "output_0": {0: "batch"},  # dynamic axes for output_0
                        "output_1": {0: "batch"},  # dynamic axes for output_1
                        "output_2": {0: "batch"},
                    },  # dynamic axes for output_2
                )
                logger.info(f"Successfully converted model to ONNX: {save_path}")
            except Exception as e:
                logger.error(f"Failed to export model to TorchScript: {e}")
        else:
            logger.info("TorchScript conversion is disabled. Skipping export.")

    def num_trainable_params(self, trainable: bool = False) -> str:
        """
        Calculate and return the number of parameters in the model
        formatted as "XX.XXM". If `trainable` is set to True, only
        the trainable parameters are counted.

        Args:
            trainable (bool): Whether to count only trainable parameters.
                            Default is False.

        Returns:
            str: Number of parameters in the format "XX.XXM".
        """

        params = self.model.parameters()
        params_to_count = (
            (p for p in params if p.requires_grad) if trainable else params
        )
        num = sum(p.numel() for p in params_to_count)

        return "{:.2f}M".format(num / 1_000_000)

    def preprocess(self, images: List[Union[str, np.ndarray]]) -> torch.Tensor:
        """
        Preprocess a list of image paths or numpy arrays, converting them to a tensor.

        Args:
        - images (List[Union[str, np.ndarray]]): List of image file paths or numpy arrays.

        Returns:
        - torch.Tensor: Tensor of preprocessed images.
        """
        tensors = []
        for item in images:
            if isinstance(item, str):
                # Item is a file path
                image = np.array(Image.open(item).convert("RGB"))
            elif isinstance(item, np.ndarray):
                # Item is a numpy array
                image = item
            else:
                raise ValueError("Unsupported image format")

            # Apply transformations and add to list
            tensors.append(test_transform(image=image)["image"])

        return torch.stack(tensors).to(self.device)

    def get_pose_details(
        self, prediction: List[int], convert_to_chinese: bool = True
    ) -> dict:
        """
        Gets details of the predicted pose and potential matches.

        Args:
        - prediction (List[int]): A list containing the predicted labels.
        - convert_to_chinese (bool): Flag to determine if the output should be in Chinese. Defaults to True.

        Returns:
        - Dict[str, Union[str, List[str]]]: A dictionary containing details of the exact match, potential matches, and gesture.
        """

        gesture_key = prediction[0]
        gesture = self.first_level_index_en.get(gesture_key, "Unknown")

        if convert_to_chinese:
            gesture = self.first_level_en_ch.get(gesture, "不確定")
            result_dict = {
                "Target": "找不到對應的瑜伽姿勢",
                "PotentialCandidate": "找不到類似的瑜伽姿勢",
                "Gesture": gesture,
            }
        else:
            result_dict = {
                "Target": "Can't match",
                "PotentialCandidate": "Can't match",
                "Gesture": gesture,
            }

        # Check for exact match
        target = next(
            (pose for pose, labels in self.poses_data.items() if prediction == labels),
            "",
        )
        if target:
            if convert_to_chinese:
                result_dict["Target"] = self.third_level_en_ch.get(target)
            else:
                result_dict["Target"] = target

        # Check for matches based on the first two labels
        partial_matches = [
            pose
            for pose, labels in self.poses_data.items()
            if prediction[:2] == labels[:2] and pose != target
        ]
        if partial_matches:
            if convert_to_chinese:
                partial_matches = [
                    self.third_level_en_ch.get(item) for item in partial_matches
                ]
            result_dict["PotentialCandidate"] = partial_matches

        return result_dict

    @torch.no_grad()
    def process(
        self, images: Union[str, np.ndarray, List[Union[str, np.ndarray]]]
    ) -> dict:
        """
        Process a list of image paths (or a single image path) to extract class probabilities and predictions.

        Args:
        - images (Union[str, List[str]]): Either a single image path or a list of image paths to be processed.

        Returns:
        - Dict[str, Dict[str, Any]]: A dictionary where the key is the image name and the value is another dictionary
                                    containing details of class confidence, class index, and combined class indices for
                                    each of the three classes.
        """
        # Extract class logits from the model
        if not isinstance(images, list):
            images = [images]
        tensor = self.preprocess(images)
        if self.onnx:
            ort_inputs = {self.model.get_inputs()[0].name: self.tensor_to_numpy(tensor)}
            # Run inference
            class_6_logic, class_20_logic, class_82_logic = self.model.run(
                None, ort_inputs
            )
            class_6_logic = torch.from_numpy(class_6_logic)
            class_20_logic = torch.from_numpy(class_20_logic)
            class_82_logic = torch.from_numpy(class_82_logic)
        else:
            class_6_logic, class_20_logic, class_82_logic = self.model(tensor)

        # Compute softmax probabilities
        class_6_prob = class_6_logic.softmax(-1)
        class_20_prob = class_20_logic.softmax(-1)
        class_82_prob = class_82_logic.softmax(-1)

        # Get max probability and class index for each class
        prob_6, class_6 = torch.max(class_6_prob, -1)
        prob_20, class_20 = torch.max(class_20_prob, -1)
        prob_82, class_82 = torch.max(class_82_prob, -1)

        # Construct the output dictionary
        output = {}
        n = 0
        for img, p_6, c_6, p_20, c_20, p_82, c_82 in zip(
            images, prob_6, class_6, prob_20, class_20, prob_82, class_82
        ):
            if not isinstance(img, str):
                img = str(n)
                n += 1
            img_name = os.path.basename(img)
            output[img_name] = {
                "first_class": {
                    "confidence": "{:.2%}".format(p_6.item()),
                    "class": c_6.item(),
                },
                "second_class": {
                    "confidence": "{:.2%}".format(p_20.item()),
                    "class": c_20.item(),
                },
                "third_class": {
                    "confidence": "{:.2%}".format(p_82.item()),
                    "class": c_82.item(),
                },
                "combine": [c_6.item(), c_20.item(), c_82.item()],
            }
        return output

    def predict(
        self,
        images: Union[str, np.ndarray, List[Union[str, np.ndarray]]],
        convert_to_chinese: bool = True,
    ) -> dict:
        """
        Predicts pose details based on the provided images.

        Args:
        - images (Union[str, List[str]]): Either a single image path or a list of image paths for prediction.
        - convert_to_chinese (bool): Flag to determine if the output should be in Chinese. Defaults to True.

        Returns:
        - Dict[str, Any]: A dictionary with image names as keys and their corresponding pose details as values.
                        If only one image is provided, returns the pose details directly.
        """

        raw_prediction = self.process(images)
        dict_prediction = {
            key: self.get_pose_details(
                value["combine"], convert_to_chinese=convert_to_chinese
            )
            for key, value in raw_prediction.items()
        }

        if len(raw_prediction) == 1:
            return next(
                iter(dict_prediction.values())
            )  # Return the sole prediction value
        else:
            return dict_prediction
