from pyannote.audio import Model
from pyannote.audio.core.task import Specifications
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
import torch.nn as nn


class RegressiveSegmentationModelMixin(Model):

    # TODO: overwrite `build` to build 3 different FC linear activations, one for each
    # TODO : we should probably overwrite the `default_activation` to return a custom activation function
    pass

class CustomSimpleSegmentationModel(RegressiveSegmentationModelMixin, SimpleSegmentationModel):
    pass

class CustomPyanNetModel(RegressiveSegmentationModelMixin, PyanNet):
    pass