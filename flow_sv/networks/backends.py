import numpy as np
from typing import Dict, Iterable, Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = [None for layer in self.layers]

        for name, layer in self.model.named_modules():
            if name in self.layers:
                layer_id = layers.index(name)
                hook = self._save_outputs_hook(layer_id)
                layer.register_forward_hook(hook)

    def _save_outputs_hook(self, layer_id: int) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
            # print(self.layers[layer_id], output.shape)
        return fn

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        _ = self.model(x)
        feats = self._features
        self._features = [None for layer in self.layers]
        return feats


def ResNet18MultiInput(n_inputs):
    # Load pretrained resnet
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    if n_inputs > 1:
        # Remember weights of the input layer
        input_sd = resnet.conv1.state_dict()
        # Replace input convolution layer
        resnet.conv1 = nn.Conv2d(
            3 * n_inputs, 64, kernel_size=7, stride=2, padding=3,
            bias=False
        )
        # Set the correct weights to the input layer
        input_sd['weight'] = torch.cat(
            [input_sd['weight']] * n_inputs,
            dim=1,
        ) / n_inputs
        noise = torch.randn_like(input_sd['weight']) * input_sd['weight'].std()
        input_sd['weight'] = (input_sd['weight'] + noise) / np.sqrt(2)
        resnet.conv1.load_state_dict(input_sd)

    return resnet


def MobileNetV2MultiInput(n_inputs):
    # Load pretrained mobilenet v2
    mobilenet = torch.hub.load(
        'pytorch/vision:v0.6.0',
        'mobilenet_v2',
        pretrained=True
    )

    if n_inputs > 1:
        # Remember weights of the input layer
        input_sd = mobilenet.features[0][0].state_dict()
        # Replace input convolution layer
        mobilenet.features[0][0] = nn.Conv2d(
            3 * n_inputs, 32, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        # Set the correct weights to the input layer
        input_sd['weight'] = torch.cat(
            [input_sd['weight']] * n_inputs,
            dim=1,
        ) / n_inputs
        noise = torch.randn_like(input_sd['weight']) * input_sd['weight'].std()
        input_sd['weight'] = (input_sd['weight'] + noise) / np.sqrt(2)
        mobilenet.features[0][0].load_state_dict(input_sd)

    return mobilenet

def ShuffleNetV2MultiInput(n_inputs):
    # Load pretrained shufflenet v2
    model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x0_5', pretrained=True)

    if n_inputs > 1:
        # Remember weights of the input layer
        input_sd = model.conv1[0].state_dict()
        # Replace input convolution layer
        model.conv1[0] = nn.Conv2d(
            3 * n_inputs, 24, kernel_size=3, stride=2, padding=1,
            bias=False
        )
        # Set the correct weights to the input layer
        input_sd['weight'] = torch.cat(
            [input_sd['weight']] * n_inputs,
            dim=1,
        ) / n_inputs
        noise = torch.randn_like(input_sd['weight']) * input_sd['weight'].std()
        input_sd['weight'] = (input_sd['weight'] + noise) / np.sqrt(2)
        model.conv1[0].load_state_dict(input_sd)

    return model
