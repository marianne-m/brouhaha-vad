import argparse
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml
from yaml.loader import SafeLoader
import pandas as pd
import numpy as np
from pyannote.audio import Model
from pyannote.audio.models.segmentation import PyanNet
from pyannote.audio.models.segmentation.debug import SimpleSegmentationModel
# from pyannote.audio.pipelines import MultiLabelSegmentation as MultilabelDetectionPipeline
# from pyannote.audio.tasks.segmentation.multilabel import MultiLabelSegmentation
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature, Segment
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
from brouhaha_vtc.models import CustomPyanNetModel, CustomSimpleSegmentationModel
from brouhaha_vtc.pipeline import RegressiveActivityDetectionPipeline

from brouhaha_vtc.task import RegressiveActivityDetectionTask
from brouhaha_vtc.utils.metrics import OptimalFScore, OptimalFScoreThreshold


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
                         'intersections': {}},
           "brouhaha": {'classes': ["A"],
                        'unions': {},
                        'intersections': {}},
           "dihard": {'classes': [f"speaker{n}" for n in range(2000)],
                        'unions': {"A": [f"speaker{n}" for n in range(2000)]},
                        'intersections': {}},
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
        vad_preprocessor = DeriveMetaLabels(**classes_kwargs)
        preprocessors = {
            "audio": FileFinder(),
            "annotation": vad_preprocessor
        }
        if args.classes == "babytrain":
            with open(Path(__file__).parent / "data/babytrain_mapping.yml") as mapping_file:
                mapping_dict = yaml.safe_load(mapping_file)["mapping"]
            preprocessors["annotation"] = ProcessorChain([
                LabelMapper(mapping_dict, keep_missing=True),
                vad_preprocessor
            ], key="annotation")
        return get_protocol(args.protocol, preprocessors=preprocessors)

    @classmethod
    def get_task(cls, args: Namespace, task_kwargs: Dict):
        protocol = cls.get_protocol(args)
        protocol.data_dir = Path(args.data_dir)
        task = RegressiveActivityDetectionTask(protocol, num_workers=8, **task_kwargs)
        task.setup()
        return task

    @classmethod 
    def get_config(cls, args:Namespace):
        config_file = args.exp_dir / "config.yaml"
        try:
            with open(config_file) as f:
                config = yaml.load(f, Loader=SafeLoader)
        except FileNotFoundError:
            print(f"The config file {config_file} was not found in {args.exp_dir}.\n"
                  f"If using the --config option, please place a config.yaml file in {args.exp_dir}")
            raise
        
        print("Using a custom config file to instantiate the model")

        return config["architecture"], config["task"]


class TrainCommand(BaseCommand):
    COMMAND = "train"
    DESCRIPTION = "train the model"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
                            default="BrouhahaDebug.SpeakerDiarization.RegressionPoetryRecital",
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
        parser.add_argument("--data_dir", type=str, required=True,
                            help="Path to the data directory")
        parser.add_argument("--gpu", type=int, default=1,
                            help="Number of gpu. Default 1.")
        parser.add_argument("--config", default=False, action="store_true",
                            help="If used, the model use the config.yml into the experimental"
                                 "folder to instantiate the model. Else, the default parameters"
                                 "are used.")

    @classmethod
    def run(cls, args: Namespace):

        model_kwargs, task_kwargs = dict(), dict()
        if args.config:
            model_kwargs, task_kwargs = cls.get_config(args)
 
        vad = cls.get_task(args, task_kwargs)

        if args.model_type == "simple":
            model = CustomSimpleSegmentationModel(task=vad)
        else:
            model = CustomPyanNetModel(task=vad, **model_kwargs)

        value_to_monitor, min_or_max = vad.val_monitor

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
                                   name="VADTest", version="", log_graph=False)
        trainer_kwargs = {'devices': args.gpu,
                          'accelerator': "gpu",
                          'callbacks': [model_checkpoint], #, early_stopping],
                          'logger': logger,
                          'max_epochs': args.epoch}
        if args.resume:
            ckpt_path = checkpoints_path / "last.ckpt"
        else:
            ckpt_path = None

        trainer = Trainer(**trainer_kwargs)
        trainer.fit(model, ckpt_path=ckpt_path)


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
        parser.add_argument("--params", type=Path,
                            help="Filename for param yaml file")
        parser.add_argument("--data_dir", type=str, required=True,
                            help="Path to the data directory")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        protocol.data_dir = Path(args.data_dir)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )

        pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
        
        fscore = OptimalFScore()
        opt_threshold = OptimalFScoreThreshold()

        for file in protocol.development():
            uri = file['uri']

            predicted = pipeline._segmentation(file)
            pred_vad = predicted[:,0]

            # get target annotation
            if not bool(file['annotation']):
                data = np.zeros(predicted.data.shape)
                annot = SlidingWindowFeature(data, predicted.sliding_window)
            else:
                annot = file['annotation'].discretize(
                    support=Segment(
                        0.0, pipeline._audio.get_duration(file)# + pipeline._segmentation.step
                    ),
                    resolution=pipeline._frames,
                ).align(to=predicted)

            f = fscore(torch.tensor(pred_vad), torch.tensor(annot.data[:,0]))
            threshold = opt_threshold(torch.tensor(pred_vad), torch.tensor(annot.data[:,0]))

            print(f'uri\t{uri}\tOptFscore\t{f}\tOptFscore threshold\t{threshold}')

        optimal_threshold = float(opt_threshold.compute())

        print(f'fscore aggregated {fscore.compute()}')
        print(f'optimal fscore threshold aggregated {optimal_threshold}')

        best_params = {
            "params": {
                "min_duration_off": 0,
                "min_duration_on": 0,
                "offset": optimal_threshold,
                "onset": optimal_threshold
            }
        }

        if args.params is None:
            params_path: Path = args.params if args.params is not None else args.exp_dir / "best_params.yml"
            with open(params_path, "w") as file:
                yaml.dump(best_params, file)


class ApplyCommand(BaseCommand):
    COMMAND = "apply"
    DESCRIPTION = "apply the model on some data"

    @classmethod
    def init_parser(cls, parser: ArgumentParser):
        parser.add_argument("-p", "--protocol", type=str,
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
        parser.add_argument("--data_dir", type=str, required=True,
                            help="Path to the data directory")
        parser.add_argument("--set", type=str, default="test",
                            help="Apply the model to this set. Possible values : dev, test, heldout. Default : test")

    @classmethod
    def run(cls, args: Namespace):

        if args.protocol:
            protocol = cls.get_protocol(args)
            protocol.data_dir = Path(args.data_dir)
            set_iter = {
                "dev": protocol.development(),
                "test": protocol.test()
            }
            data_iterator = set_iter[args.set]
        else:
            def iter():
                files = []
                files.extend(Path(args.data_dir).glob("*.wav"))
                files.extend(Path(args.data_dir).glob("**/*.wav"))
                for file in files:
                    yield {
                        "uri": file.stem,
                        "audio": file
                    }
            data_iterator = iter()

        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        pipeline = RegressiveActivityDetectionPipeline(segmentation=model)

        if args.params is None:
            params_path: Path = args.params if args.params is not None else args.exp_dir / "best_params.yml"
        else:
            params_path = Path(args.params)
        pipeline.load_params(params_path)

        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder
        apply_folder.mkdir(parents=True, exist_ok=True)

        rttm_folder = apply_folder / "rttm_files"
        snr_folder = apply_folder / "detailed_snr_labels"
        c50_folder = apply_folder / "c50"

        rttm_folder.mkdir(parents=True, exist_ok=True)
        snr_folder.mkdir(parents=True, exist_ok=True)
        c50_folder.mkdir(parents=True, exist_ok=True)

        for file in tqdm(list(data_iterator)):
            logging.info(f"Inference for file {file['uri']}")
            inference = pipeline(file)
            annotation: Annotation = inference["annotation"]
            snr = inference["snr"]
            c50 = inference["c50"]
            with open(rttm_folder / (file["uri"].replace("/", "_") + ".rttm"), "w") as rttm_file:
                annotation.write_rttm(rttm_file)
            with open(snr_folder / (file["uri"].replace("/", "_") + ".npy"), "wb") as snr_file:
                np.save(snr_file, snr)
            with open(c50_folder / (file["uri"].replace("/", "_") + ".npy"), "wb") as c50_file:
                np.save(c50_file, c50)
            with open(apply_folder / "reverb_labels.txt", "a") as label_file:
                label_file.write(f"{file['uri']} {np.mean(c50)}\n")
            with open(apply_folder / "mean_snr_labels.txt", "a") as snr_file:
                snr_file.write(f"{file['uri']} {np.mean(snr)}\n")


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
        parser.add_argument("--data_dir", type=str, required=True,
                            help="Path to the data directory")
        parser.add_argument("--set", type=str, default="test",
                            help="Apply the model to this set. Possible values : dev, test, heldout. Default : test")

    @classmethod
    def run(cls, args: Namespace):
        protocol = cls.get_protocol(args)
        protocol.data_dir = Path(args.data_dir)

        apply_folder: Path = args.exp_dir / "apply/" if args.apply_folder is None else args.apply_folder

        rttm_folder = apply_folder / "rttm_files"
        snr_folder = apply_folder / "detailed_snr_labels"
        c50_file = apply_folder / "reverb_labels.txt"

        annotations: Dict[str, Annotation] = {}
        for filepath in rttm_folder.glob("*.rttm"):
            rttm_annots = load_rttm(filepath)
            annotations.update(rttm_annots)
        model = Model.from_pretrained(
            Path(args.model_path),
            strict=False,
        )
        pipeline = RegressiveActivityDetectionPipeline(segmentation=model)
        metric: BaseMetric = pipeline.get_metric()

        filenames = []
        snr_test_metric = []
        c50_test_metric = []
        c50_df = pd.read_csv(c50_file, sep=" ", header=None)
        c50 = {key: val for key, val in zip(c50_df[0], c50_df[1])}

        set_iter = {
            "dev": protocol.development(),
            "test": protocol.test()
        }

        for file in set_iter[args.set]:
            if file["uri"] not in annotations.keys():
                continue
            filenames.append(file["uri"])
            # score vad
            inference = annotations[file["uri"]]
            metric["vadTestMetric"](file["annotation"], inference, file["annotated"])

            # score snr
            # get predicted snr
            snr_preds_file = str(snr_folder / file['uri']) + ".npy"
            preds_array = np.load(snr_preds_file)
            preds_array = np.expand_dims(preds_array, axis=1)

            resolution = file['annotated'][0].duration / preds_array.shape[0]
            sliding_window = SlidingWindow(
                start=0, step=resolution, duration=resolution
            )
            preds = SlidingWindowFeature(preds_array, sliding_window)

            # get target snr
            target = file['target_features']['snr']
            target = target.align(to=preds)

            # get target annotation to mask snr
            annot = file["annotation"].discretize(
                support=file['annotated'][0], resolution=resolution, duration=file['annotated'][0].duration
            )

            mse_snr = metric["snrTestMetric"](torch.tensor(preds.data), torch.tensor(target.data), weights=torch.Tensor(annot.data))
            snr_test_metric.append(float(mse_snr))

            # score c50
            c50_pred = c50[file["uri"]]
            c50_target = file["target_features"]["c50"]

            mse_c50 = metric["c50TestMetric"](torch.tensor(c50_pred), torch.tensor(c50_target))
            c50_test_metric.append(float(mse_c50))

        # totals for snr and c50
        filenames.append("TOTAL")
        snr_test_metric.append(float(metric["snrTestMetric"].compute()))
        c50_test_metric.append(float(metric["c50TestMetric"].compute()))

        df_fscore: pd.DataFrame = metric["vadTestMetric"].report(display=True)
        df_snr_c50 = pd.DataFrame({"uri": filenames, "MSE(snr)": snr_test_metric, "MSE(c50)": c50_test_metric})
        if args.report_path is not None:
            args.report_path.mkdir(parents=True, exist_ok=True)
            df_fscore.to_csv(args.report_path / "fscore.csv")
            df_snr_c50.to_csv(args.report_path / "snr_c50_scores.csv")


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
