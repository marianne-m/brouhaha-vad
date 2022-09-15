import tempfile
from copy import deepcopy
from types import MethodType
from typing import Optional, Union, Callable

import numpy as np
import yaml
from pytorch_lightning import Trainer
from torch.optim import SGD
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference
from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineAugmentation,
    PipelineInference,
    PipelineModel,
    get_augmentation,
    get_devices,
    get_inference,
    get_model,
)
from pyannote.audio.tasks import VoiceActivityDetection as VoiceActivityDetectionTask
from pyannote.audio.utils.signal import Binarize
from pyannote.audio.utils.loss import mse_loss
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
)
from pyannote.pipeline.parameter import Categorical, Integer, LogUniform, Uniform



class RegressiveActivityDetectionPipeline(Pipeline):
    """Voice activity detection pipeline

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation (or voice activity detection) model.
        Defaults to "pyannote/segmentation".
        See pyannote.audio.pipelines.utils.get_model for supported format.
    fscore : bool, optional
        Optimize (precision/recall) fscore. Defaults to optimizing detection
        error rate.
    inference_kwargs : dict, optional
        Keywords arguments passed to Inference.

    Hyper-parameters
    ----------------
    onset, offset : float
        Onset/offset detection thresholds
    min_duration_on : float
        Remove speech regions shorter than that many seconds.
    min_duration_off : float
        Fill non-speech regions shorter than that many seconds.
    """

    def __init__(
        self,
        segmentation: PipelineModel = None,
        **inference_kwargs,
    ):
        super().__init__()

        self.segmentation = segmentation

        # load model and send it to GPU (when available and not already on GPU)
        model = get_model(segmentation)
        if model.device.type == "cpu":
            (segmentation_device,) = get_devices(needs=1)
            model.to(segmentation_device)

        # inference_kwargs["pre_aggregation_hook"] = lambda scores: np.max(
        #     scores, axis=-1, keepdims=True
        # )
        self._segmentation = Inference(model, **inference_kwargs)

        #  hyper-parameters used for hysteresis thresholding
        self.onset = Uniform(0.0, 1.0)
        self.offset = Uniform(0.0, 1.0)

        # hyper-parameters used for post-processing i.e. removing short speech regions
        # or filling short gaps between speech regions
        self.min_duration_on = Uniform(0.0, 1.0)
        self.min_duration_off = Uniform(0.0, 1.0)

        # threshold for regression
        self.snr_threshold = 0.5
        self.c50_threshold = 0.5

    def default_parameters(self):
        return {
            "onset": 0.800,
            "offset": 0.800,
            "min_duration_on": 0.136,
            "min_duration_off": 0.067,
        }

    def get_parameters(self, threshold_file, epoch):
        file = self.model.logger.log_dir + "/vad_fscore_threshold.yaml"
        with open(threshold_file, "r") as file:
            opt_threshold = yaml.load(file, Loader=yaml.loader.SafeLoader)
        threshold = opt_threshold[f"epoch_{epoch}"]["optimal_th"]
        params = {
            "onset": threshold,
            "offset": threshold,
            "min_duration_on": 0.136,
            "min_duration_off": 0.067,
        }
        return params

    def classes(self):
        return ["SPEECH"]

    def initialize(self):
        """Initialize pipeline with current set of parameters"""

        self._binarize = Binarize(
            onset=self.onset,
            offset=self.offset,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )

    CACHED_SEGMENTATION = "cache/segmentation/inference"

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> Annotation:
        """Apply voice activity detection

        Parameters
        ----------
        file : AudioFile
            Processed file.
        hook : callable, optional
            Hook called after each major step of the pipeline with the following
            signature: hook("step_name", step_artefact, file=file)

        Returns
        -------
        speech : Annotation
            Speech regions.
        """

        # setup hook (e.g. for debugging purposes)
        hook = self.setup_hook(file, hook=hook)

        # apply segmentation model (only if needed)
        # output shape is (num_chunks, num_frames, 1)
        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations: SlidingWindowFeature = self._segmentation(file)

        hook("segmentation", segmentations)

        speech_seg = SlidingWindowFeature(segmentations.data, segmentations.sliding_window)
        speech_seg.data = np.expand_dims(speech_seg.data[:,0], axis=1)
        
        speech: Annotation = self._binarize(speech_seg)
        speech.uri = file["uri"]

        snr_labels = segmentations.data[:, 1]
        c50_labels = segmentations.data[:, 2]

        return {
            "annotation": speech.rename_labels({label: "A" for label in speech.labels()}),
            "snr": snr_labels,
            "c50": c50_labels
        }

    def get_metric(self) -> Union[DetectionErrorRate, DetectionPrecisionRecallFMeasure]:
        """Return new instance of detection metric"""

        return {
            "vadTestMetric": DetectionPrecisionRecallFMeasure(collar=0.0, skip_overlap=False),
            "snrTestMetric": mse_loss,
            "c50TestMetric": mse_loss
        }
