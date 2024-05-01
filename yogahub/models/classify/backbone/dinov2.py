from enum import Enum
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from loguru import logger
from torch import Tensor

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"


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


def _make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"


class Weights(Enum):
    IMAGENET1K = "IMAGENET1K"


def _make_dinov2_linear_classification_head(
    *,
    model_name: str = "dinov2_vitl14",
    embed_dim: int = 1024,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    **kwargs,
):
    if layers not in (1, 4):
        raise AssertionError(f"Unsupported number of layers: {layers}")
    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    linear_head = nn.Linear((1 + layers) * embed_dim, 1_000)

    if pretrained:
        layers_str = str(layers) if layers == 4 else ""
        url = (
            _DINOV2_BASE_URL + f"/{model_name}/{model_name}_linear{layers_str}_head.pth"
        )
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        linear_head.load_state_dict(state_dict, strict=False)

    return linear_head


class _LinearClassifierWrapperA(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        linear_head: nn.Module,
        layers: int = 4,
        multiclassifier: List[int] = [],
    ):
        super().__init__()
        self.backbone = backbone
        self.linear_head = linear_head
        self.layers = layers
        self.multiclassifier = nn.ModuleList(
            [
                Mlp(self.linear_head.out_features, out_features=num_class, drop=0.3)
                for num_class in multiclassifier
            ]
        )

    def forward(self, x):
        if self.layers == 1:
            x = self.backbone.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"

        x = self.linear_head(linear_input)

        if self.multiclassifier:
            return [layer(x) for layer in self.multiclassifier]

        return x


class _LinearClassifierWrapperB(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        multiclassifier: List[int] = [],
    ):
        super().__init__()
        self.backbone = backbone
        out_dimension = self.backbone.blocks[-1].mlp.fc2.out_features
        self.multiclassifier = nn.ModuleList(
            [
                Mlp(out_dimension * 2, out_features=num_class, drop=0.3)
                for num_class in multiclassifier
            ]
        )

    def forward(self, x):
        x = self.backbone.get_intermediate_layers(x, n=4, return_class_token=True)
        linear_input_6 = torch.cat(
            [
                x[0][1],
                x[0][0].mean(dim=1),
            ],
            dim=1,
        )
        linear_input_20 = torch.cat(
            [
                x[2][1],
                x[2][0].mean(dim=1),
            ],
            dim=1,
        )
        linear_input_82 = torch.cat(
            [
                x[3][1],
                x[3][0].mean(dim=1),
            ],
            dim=1,
        )
        linear_input = [linear_input_6, linear_input_20, linear_input_82]
        # fmt: on
        output = []
        for input_tensor, layer in zip(linear_input, self.multiclassifier):
            output.append(layer(input_tensor))
        return output


def _make_dinov2_linear_classifier(
    *,
    arch_name: str = "vit_base",
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    multiclassifier: List[int] = [],
    **kwargs,
):
    if arch_name == "dino2_vit_base":
        backbone_name = "dinov2_vitb14"
    if arch_name == "dino2_vit_small":
        backbone_name = "dinov2_vits14"
    logger.info(f"Activate dinov2 with backbone: {backbone_name}")
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, **kwargs)

    """
    embed_dim = backbone.embed_dim
    patch_size = backbone.patch_size
    model_name = _make_dinov2_model_name(arch_name, patch_size)
    linear_head = _make_dinov2_linear_classification_head(
    model_name=model_name,
    embed_dim=embed_dim,
    layers=layers,
    pretrained=pretrained,
    weights=weights,
    )
    return _LinearClassifierWrapperA(
        backbone=backbone,
        linear_head=linear_head,
        layers=layers,
        multiclassifier=multiclassifier,
    )
    """
    return _LinearClassifierWrapperB(backbone=backbone, multiclassifier=multiclassifier)


def dinov2_vitb14_lc(
    *,
    layers: int = 4,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.IMAGENET1K,
    multiclassifier=[],
    version="vit_base",
    **kwargs,
):
    """
    Linear classifier (1 or 4 layers) on top of a DINOv2 ViT-B/14 backbone (optionally) pretrained on the LVD-142M dataset and trained on ImageNet-1k.
    """
    return _make_dinov2_linear_classifier(
        arch_name=version,
        layers=layers,
        pretrained=pretrained,
        weights=weights,
        multiclassifier=multiclassifier,
        **kwargs,
    )
