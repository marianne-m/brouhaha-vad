import argparse
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml
import pandas as pd
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
from pyannote.audio.pipelines import MultilabelDetection as MultilabelDetectionPipeline
from pyannote.audio.tasks.segmentation.multilabel_detection import MultilabelDetection
from pyannote.core import Annotation
from pyannote.audio.utils.preprocessors import DeriveMetaLabels
from pyannote.database import FileFinder, get_protocol, ProtocolFile
from pyannote.database.protocol.protocol import Preprocessor
from pyannote.database.util import load_rttm, LabelMapper
from pyannote.metrics.base import BaseMetric
from pyannote.pipeline import Optimizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm


class ProcessorChain:

    def __init__(self, preprocessors: List[Preprocessor], key: str):
        self.procs = preprocessors
        self.key = key

    def __call__(self, file: ProtocolFile):
        file_cp: Dict[str, Any] = abs(file)
        for proc in self.procs:
            out = proc(file_cp)
            file_cp[self.key] = out

        return out


DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

CLASSES = {"vtcdebug": {'classes': ["READER", "AGREER", "DISAGREER"],
                        'unions': {"COMMENTERS": ["AGREER", "DISAGREER"]},
                        'intersections': {}},
           "basal_voice": {'classes': ["P", "NP"],
                           'unions': {},
                           'intersections': {}},
           "babytrain": {'classes': ["MAL", "FEM", "CHI", "KCHI"],
                         'unions': {"SPEECH": ["MAL", "FEM", "CHI", "KCHI"]},
                         'intersections': {}}
           }


class BaseCommand:
    COMMAND = "command"
    DESCRIPTION = "Command description"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        pass

    @classmethod
    def run(cls, args: Namespace):
        pass

    @classmethod
    def get_protocol(cls, args: Namespace):
        classes_kwargs = CLASSES[args.classes]
        vtc_preprocessor = DeriveMetaLabels(**classes_kwargs)
        preprocessors = {
            "audio": FileFinder(),
            "annotation": vtc_preprocessor
        }
        if args.classes == "babytrain":
            with open(Path(__file__).parent / "data/babytrain_mapping.yml") as mapping_file:
                mapping_dict = yaml.safe_load(mapping_file)["mapping"]
            preprocessors["annotation"] = ProcessorChain([
                LabelMapper(mapping_dict, keep_missing=True),
                vtc_preprocessor
            ], key="annotation")
        return get_protocol(args.protocol, preprocessors=preprocessors)

    @classmethod
    def get_task(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        task = MultilabelDetection(protocol, duration=2.00)
        task.setup()
        return task


class TrainCommand(BaseCommand):
    COMMAND = "train"
    DESCRIPTION = "train the model"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            required=True,
                            type=str, help="Model architecture")
        parser.add_argument("--model_type", choices=["simple", "pyannet"],
                            required=True,
                            type=str, help="Model model checkpoint")
        parser.add_argument("--resume", action="store_true",
                            help="Resume from last checkpoint")
        parser.add_argument("--epoch", type=int, required=True,
                            help="Number of train epoch")

    @classmethod
    def run(cls, args: Namespace):

        vtc = cls.get_task(args)

        if args.model_type == "simple":
            model = SimpleSegmentationModel(task=vtc)
        else:
            model = PyanNet(task=vtc)

        value_to_monitor, min_or_max = vtc.val_monitor

        checkpoints_path: Path = args.exp_dir / "checkpoints/"
        checkpoints_path.mkdir(parents=True, exist_ok=True)

        checkpoints_kwargs = {
            'monitor': value_to_monitor,
            'mode': min_or_max,
            'save_top_k': 5,
            'every_n_epochs': 1,
            'save_last': True,
            'dirpath': checkpoints_path,
            'filename': f"{{epoch}}-{{{value_to_monitor}:.6f}}",
            'verbose': True}

        model_checkpoint = ModelCheckpoint(**checkpoints_kwargs)

        early_stopping = EarlyStopping(
            monitor=value_to_monitor,
            mode=min_or_max,
            min_delta=0.0,
            patience=10,
            strict=True,
            verbose=False)

        logger = TensorBoardLogger(args.exp_dir,
                                   name="VTCTest", version="", log_graph=False)
        trainer_kwargs = {'devices': 1,
                          'accelerator': "gpu",
                          'callbacks': [model_checkpoint, early_stopping],
                          'logger': logger}
        if args.resume:
            trainer_kwargs["resume_from_checkpoint"] = checkpoints_path / "last.ckpt"

        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model)


class TuneCommand(BaseCommand):
    COMMAND = "tune"
    DESCRIPTION = "tune the model hyperparameters using optuna"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            required=True,
                            type=str, help="Model model checkpoint")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Model checkpoint to tune pipeline with")
        parser.add_argument("-nit", "--n_iterations", type=int, default=50,
                            help="Number of tuning iterations")
        parser.add_argument("--metric", choices=["fscore", "ier"],
                            default="fscore")
        parser.add_argument("--params", type=Path, default=Path("best_params.yml"),
                            help="Filename for param yaml file")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        # Dirty fix for the non-serialization of the task params
        pipeline = MultilabelDetectionPipeline(segmentation=model,
                                               fscore=args.metric == "fscore")
        # pipeline.instantiate(pipeline.default_parameters())
        validation_files = list(protocol.development())
        optimizer = Optimizer(pipeline)
        optimizer.tune(validation_files,
                       n_iterations=args.n_iterations,
                       show_progress=True)
        best_params = optimizer.best_params
        logging.info(f"Best params: \n{best_params}")
        params_filepath: Path = args.exp_dir / args.params
        logging.info(f"Saving params to {params_filepath}")
        pipeline.instantiate(best_params)
        pipeline.dump_params(params_filepath)


class ApplyCommand(BaseCommand):
    COMMAND = "apply"
    DESCRIPTION = "apply the model on some data"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            required=True,
                            type=str, help="Model model checkpoint")
        parser.add_argument("-m", "--model_path", type=Path, required=True,
                            help="Model checkpoint to run pipeline with")
        parser.add_argument("--params", type=Path,
                            help="Path to best params. Default to EXP_DIR/best_params.yml")
        parser.add_argument("--apply_folder", type=Path,
                            help="Path to apply folder")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        pipeline = MultilabelDetectionPipeline(segmentation=model)
        params_path: Path = args.params if args.params is not None else args.exp_dir / "best_params.yml"
        pipeline.load_params(params_path)
        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        apply_folder.mkdir(parents=True, exist_ok=True)

        for file in tqdm(list(protocol.test())):
            logging.info(f"Inference for file {file['uri']}")
            annotation: Annotation = pipeline(file)
            with open(apply_folder / (file["uri"].replace("/", "_") + ".rttm"), "w") as rttm_file:
                annotation.write_rttm(rttm_file)


class ScoreCommand(BaseCommand):
    COMMAND = "score"
    DESCRIPTION = "score some inference"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="VTCDebug.SpeakerDiarization.PoetryRecitalDiarization",
                            help="Pyannote database")
        parser.add_argument("--apply_folder", type=Path,
                            help="Path to the inference files")
        parser.add_argument("--classes", choices=CLASSES.keys(),
                            required=True,
                            type=str, help="Model architecture")
        parser.add_argument("--metric", choices=["fscore", "ier"],
                            default="fscore")
        parser.add_argument("--model_path", type=Path, required=True,
                            help="Model model checkpoint")
        parser.add_argument("--report_path", type=Path, required=True,
                            help="Path to report csv")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        annotations: Dict[str, Annotation] = {}
        for filepath in apply_folder.glob("*.rttm"):
            rttm_annots = load_rttm(filepath)
            annotations.update(rttm_annots)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        pipeline = MultilabelDetectionPipeline(segmentation=model,
                                               fscore=args.metric == "fscore")
        metric: BaseMetric = pipeline.get_metric()

        for file in protocol.test():
            if file["uri"] not in annotations:
                continue
            inference = annotations[file["uri"]]
            metric(file["annotation"], inference, file["annotated"])

        df: pd.DataFrame = metric.report(display=True)
        if args.report_path is not None:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.report_path)


commands = [TrainCommand, TuneCommand, ApplyCommand, ScoreCommand]

argparser = argparse.ArgumentParser()
argparser.add_argument("-v", "--verbose", action="store_true",
                       help="Show debug information in the standard output")
argparser.add_argument("exp_dir", type=Path,
                       help="Experimental folder")
subparsers = argparser.add_subparsers()

for command in commands:
    subparser = subparsers.add_parser(command.COMMAND)
    subparser.set_defaults(func=command.run,
                           command_class=command,
                           subparser=subparser)
    command.init_parser(subparser)

if __name__ == '__main__':
    args = argparser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if hasattr(args, "func"):
        args.func(args)
    else:
        argparser.print_help()
