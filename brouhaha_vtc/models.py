from turtle import forward
from typing import Optional
from pyannote.audio import Model
from pyannote.audio.core.task import Specifications
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.core.task import Task

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParametricSigmoid(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x: torch.Tensor):
        return (self.beta - self.alpha) * torch.special.expit(x) + self.alpha


class CustomClassifier(nn.Module):
    def __init__(self, in_features, out_features: int) -> None:
        super().__init__()
        self.linears = nn.ModuleDict({
            'vad': nn.Linear(in_features, out_features),
            'snr': nn.Linear(in_features, 1),
            'c50': nn.Linear(in_features, 1),
        })
    
    def forward(self, x: torch.Tensor):
        out = dict()
        for mode, linear in self.linears.items():
            _output = linear(x) 
            out[mode] = _output
        
        return out


class CustomActivation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activations = nn.ModuleDict({
            'vad': nn.Sigmoid(),
            'snr': ParametricSigmoid(30, -10),
            'c50': ParametricSigmoid(60, -10),
        })

    def forward(self, x: torch.Tensor):
        out = list()
        for mode, activation in self.activations.items():
            _output = activation(x[mode]) 
            out.append(_output)

        out = torch.stack(out)
        out = rearrange(out, "n b t o -> b t (n o)")
        return out


class RegressiveSegmentationModelMixin(Model):
    pass

    # TODO: overwrite `build` to build 3 different FC linear activations, one for each
    # TODO : we should probably overwrite the `default_activation` to return a custom activation function
    
        #TODO: Create a linear + activation block for the three last linear layers
        # The aim is to modify build and not forward
        # dict module for linear blocks


    def build(self):
        self.classifier = CustomClassifier(32 * 2, len(self.specifications.classes))
        self.activation = CustomActivation()
    

class CustomSimpleSegmentationModel(RegressiveSegmentationModelMixin, SimpleSegmentationModel):
    pass


class CustomPyanNetModel(RegressiveSegmentationModelMixin, PyanNet):
    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.lstm["hidden_size"] * (
                2 if self.hparams.lstm["bidirectional"] else 1
            )

        self.classifier = CustomClassifier(in_features, len(self.specifications.classes))
        self.activation = CustomActivation()