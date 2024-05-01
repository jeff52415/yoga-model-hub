import logging
from typing import Callable, List, Optional

import timm
import torch
import torch.nn as nn
from torch import Tensor

# Configure logging to write messages to a file (app.log) and the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class TimmModelWrapper(nn.Module):
    model_layer_config = {
        # "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384": [256, 512, 1024],
        "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384": [512, 512, 1024],
        # "convnext_small.in12k_ft_in1k_384": [192, 384, 768],
        "convnext_small.in12k_ft_in1k_384": [384, 384, 768],
        # "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k": [256, 512, 1024],
        "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k": [512, 512, 1024],
    }

    def __init__(
        self,
        timm_model: str = "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
        multiclassifier: List[int] = [6, 20, 82],
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.5,
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()
        # avail_pretrained_models = timm.list_models(pretrained=True,)
        if timm_model == "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k":
            logger.warning(f"Input image size have to be 384 * 384")

        self.backbone = timm.create_model(
            timm_model,
            pretrained=pretrained,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            features_only=True,
            out_indices=[-2, -1],
            **kwargs,
        )
        self.adaptive_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.multiclassifier = nn.ModuleList([])
        layer_size = TimmModelWrapper.model_layer_config.get(timm_model)

        for num_class, size in zip(multiclassifier, layer_size):
            self.multiclassifier.append(
                Mlp(size, out_features=num_class, drop=drop_rate)
            )

        num = self.count_parameters()
        logger.info(f"Use bakcbone: {timm_model}, Total parameters: {num}")

    def count_parameters(
        self,
    ) -> int:
        """Counts the number of trainable parameters in a PyTorch model.

        Args:
            model (nn.Module): The model to count parameters for.

        Returns:
            int: The number of trainable parameters.
        """
        num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num = "{:.2f}M".format(num / 1_000_000)
        return num

    # @torch.jit.script_method
    def forward(self, x):
        batch = x.shape[0]
        stack_layers = self.backbone(x)
        outputs: List[torch.Tensor] = []

        # Determine the index at which to switch from stack_layers[0] to stack_layers[1]
        switch_index = len(self.multiclassifier) - len(stack_layers) + 1

        for i, layer in enumerate(self.multiclassifier):
            # Choose the tensor from stack_layers based on the current index
            tensor_index = 0 if i < switch_index else 1
            tensor = self.adaptive_pooling(stack_layers[tensor_index]).reshape(
                batch, -1
            )
            tensor = layer(tensor)
            outputs.append(tensor)
        # outputs = tuple(outputs)
        return outputs[0], outputs[1], outputs[2]
