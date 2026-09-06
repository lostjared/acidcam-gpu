#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
from torch import nn


LAYER_SPECS = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu4_1": 18,
    "relu4_2": 20,
    "relu4_3": 22,
    "relu5_1": 25,
    "relu5_2": 27,
    "relu5_3": 29,
}
DEFAULT_LAYERS = list(LAYER_SPECS)


class DreamFeatureExtractor(nn.Module):
    acmxvk_deep_dream_format: str
    acmxvk_deep_dream_version: int
    architecture: str
    default_layer: str
    input_channels: int
    minimum_input_size: int
    input_mean: List[float]
    input_std: List[float]
    layer_names: List[str]
    layer_source_indices: List[int]

    def __init__(
        self,
        features: nn.Sequential,
        layer_names: List[str],
        layer_source_indices: List[int],
        default_layer: str,
    ) -> None:
        super().__init__()
        self.features = features
        self.acmxvk_deep_dream_format = "acmxvk-deep-dream"
        self.acmxvk_deep_dream_version = 1
        self.architecture = "vgg16"
        self.default_layer = default_layer
        self.input_channels = 3
        self.minimum_input_size = 32
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.layer_names = layer_names
        self.layer_source_indices = layer_source_indices

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        outputs = torch.jit.annotate(List[torch.Tensor], [])
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in self.layer_source_indices:
                outputs.append(image)
        return outputs


def parse_layers(value: str) -> tuple[List[str], List[int]]:
    requested = [entry.strip() for entry in value.split(",") if entry.strip()]
    if not requested:
        raise ValueError("at least one feature layer is required")
    unknown = sorted(set(requested) - LAYER_SPECS.keys())
    if unknown:
        raise ValueError(f"unknown VGG16 feature layer(s): {', '.join(unknown)}")
    if len(requested) != len(set(requested)):
        raise ValueError("feature layers must not be repeated")
    ordered = sorted(requested, key=LAYER_SPECS.__getitem__)
    return ordered, [LAYER_SPECS[name] for name in ordered]


def build_model(
    layer_names: List[str], layer_indices: List[int], default_layer: str, weights: str
) -> DreamFeatureExtractor:
    try:
        from torchvision.models import VGG16_Weights, vgg16
    except ImportError as error:
        raise RuntimeError(
            "torchvision is required; on Arch Linux install "
            "python-torchvision-cuda"
        ) from error

    selected_weights = VGG16_Weights.DEFAULT if weights == "default" else None
    source = vgg16(weights=selected_weights).features.eval()
    maximum_index = layer_indices[-1]
    features = nn.Sequential(*list(source.children())[: maximum_index + 1])
    model = DreamFeatureExtractor(
        features, layer_names, layer_indices, default_layer
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def metadata(
    layer_names: List[str], layer_indices: List[int], default_layer: str, weights: str
) -> dict:
    return {
        "format": "acmxvk-deep-dream",
        "version": 1,
        "architecture": "vgg16",
        "weights": weights,
        "default_layer": default_layer,
        "input": {
            "channels": 3,
            "minimum_size": 32,
            "color": "RGB",
            "range": [0.0, 1.0],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "layers": [
            {"name": name, "source_index": index}
            for name, index in zip(layer_names, layer_indices)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a VGG16 feature model for ACMXVK Deep Dream"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--layers", default=",".join(DEFAULT_LAYERS), help="comma-separated layer names"
    )
    parser.add_argument("--default-layer", default="relu4_2")
    parser.add_argument("--weights", choices=("default", "none"), default="default")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    try:
        layer_names, layer_indices = parse_layers(arguments.layers)
        if arguments.default_layer not in layer_names:
            raise ValueError("--default-layer must be included in --layers")
    except ValueError as error:
        parser.error(str(error))

    output = arguments.output.expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".json")
    if not arguments.force and (output.exists() or sidecar.exists()):
        parser.error("output already exists; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(
        layer_names, layer_indices, arguments.default_layer, arguments.weights
    )
    scripted = torch.jit.script(model)
    metadata_text = json.dumps(
        metadata(layer_names, layer_indices, arguments.default_layer, arguments.weights),
        indent=2,
    )
    temporary_model = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp-{os.getpid()}")
    try:
        torch.jit.save(
            scripted,
            str(temporary_model),
            _extra_files={"acmxvk_deep_dream.json": metadata_text},
        )
        temporary_sidecar.write_text(metadata_text + "\n", encoding="utf-8")
        temporary_sidecar.replace(sidecar)
        temporary_model.replace(output)
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_sidecar.unlink(missing_ok=True)

    print(f"Exported ACMXVK Deep Dream model: {output}")
    print(f"Metadata: {sidecar}")
    print(f"Layers: {', '.join(layer_names)}")
    print(f"Default layer: {arguments.default_layer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
