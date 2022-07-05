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


class RegressiveSegmentationModelMixin(Model):
    pass

    # TODO: overwrite `build` to build 3 different FC linear activations, one for each
    # TODO : we should probably overwrite the `default_activation` to return a custom activation function
    
        #TODO: Create a linear + activation block for the three last linear layers
        # The aim is to modify build and not forward
        # dict module for linear blocks


    def build(self):
        self.choices = nn.ModuleDict({
                'vad': nn.Linear(32 * 2, len(self.specifications.classes)),
                'snr': nn.Linear(32 * 2, 1),
                'c50': nn.Linear(32 * 2, 1),
        })

        self.activations = nn.ModuleDict({
                'vad': nn.Sigmoid(),
                'snr': ParametricSigmoid(30, -10),
                'c50': ParametricSigmoid(60, -10),
        })


    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        waveforms : (batch, time, channel)

        Returns
        -------
        scores : (batch, time, [classes snr c50])
        """
        # extract MFCC
        mfcc = self.mfcc(waveforms)
        # pass MFCC sequeence into the recurrent layer
        output, hidden = self.lstm(rearrange(mfcc, "b c f t -> b t (c f)"))
        # apply the final classifier to get logits
        out = []

        for mode in ['vad', 'snr', 'c50']:
            _output = self.activations[mode](self.choices[mode](output)) 
            out.append(_output)

        out = torch.stack(out)
        out = rearrange(out, "n b t o -> b t (n o)")
        return out


class CustomSimpleSegmentationModel(RegressiveSegmentationModelMixin, SimpleSegmentationModel):
    pass

class CustomPyanNetModel(RegressiveSegmentationModelMixin, PyanNet):
    pass