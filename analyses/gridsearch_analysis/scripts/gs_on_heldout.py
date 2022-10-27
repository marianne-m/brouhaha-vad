# This script gets the config settings and the metrics by epoch
# and returns a csv file with all models with config params and best epoch 
# regarding to the validation metric
# 
# usage :
#   python get_data_for_gs_analysis.py path/to/all/models
#
# example :
#   python get_data_for_gs_analysis.py /gpfsscratch/rech/xdz/commun/Brouhaha_GridSearch
#

from cmath import exp
from attr import has
import numpy as np
import pandas as pd
import argparse
import yaml
import sys
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Union

SNR_MIN = -15
SNR_MAX = 80
C50_MIN = -10
C50_MAX = 60
MAX_ERROR_SNR = SNR_MAX - SNR_MIN
MAX_ERROR_C50 = C50_MAX - C50_MIN




METRICS = {
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-c50ValMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-snrValMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-vadOptiTh",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-vadValMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-ValidationMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-TrainLoss",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-vadLoss",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-snrLoss",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-c50Loss",
    "epoch"
}
MAPPER = {
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-c50ValMetric": "c50ValMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-snrValMetric": "snrValMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-vadOptiTh": "vadOptiTh",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-vadValMetric": "vadValMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-ValidationMetric": "ValidationMetric",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-TrainLoss": "TrainLoss",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-vadLoss": "vadLoss",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-snrLoss": "snrLoss",
    "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-c50Loss": "c50Loss"
}
VALMETRIC = "RegressiveActivityDetectionTask-BrouhahaSpeakerDiarizationNoisySpeakerDiarization-ValidationMetric"



def prepare_data_dir(
    heldout_dir: Union[Path, str],
    experiment: Union[Path, str],
    best: int,
    gridsearch_dir: Union[Path, str]
) -> None:
    """
    """
    exp_dir = Path(heldout_dir) / Path(experiment)
    exp_dir.mkdir(exist_ok=True, parents=True)

    checkpoint_path = exp_dir / "checkpoints"
    checkpoint_path.mkdir(exist_ok=True)

    original_checkpoint_path = Path(gridsearch_dir) / Path(experiment) / "checkpoints"
    best_ckpt = list(original_checkpoint_path.glob(f"epoch={int(best)}*"))[0]

    shutil.copy(original_checkpoint_path / best_ckpt, checkpoint_path / "best.ckpt")
    shutil.copy(Path(gridsearch_dir) / Path(experiment) / "config.yaml", exp_dir / "config.yaml")


def get_metrics(
    exp_dir: Union[Path, str]
) -> dict:
    """
    Get metrics from results directory
    """
    result_dir = Path(exp_dir) / "results"

    with open(result_dir / "fscore.csv", "r") as file:
        total_fscore = file.readlines()[-1]
    fscore = float(total_fscore.split(',')[1])

    with open(result_dir / "snr_c50_scores.csv", "r") as file:
        total_snr_c50 = file.readlines()[-1]
    _, _, snr, c50 = total_snr_c50.split(',')
    
    snr = float(snr)
    c50 = float(c50)

    validation = ((1 - fscore/100) + snr / MAX_ERROR_SNR + c50 / MAX_ERROR_C50) / 3

    return {
        "snrValMetric": snr,
        "c50ValMetric": c50,
        "vadValMetric": fscore,
        "ValidationMetric": validation
    }


def save_data(
    data: dict,
    path: Path
) -> None:
    """
    Save data to a yaml file
    """
    try:
        with open(path / "metrics.yaml", "w") as file:
            yaml.dump(data, file)
    except PermissionError:
        print(f"Cannot save the metrics.yaml file in {path} - permission denied")

def get_metrics_data(
    path_to_metrics: Path
) -> dict:
    """
    Save data to a yaml file
    """
    with open(path_to_metrics, "r") as file:
        metrics_data = yaml.load(file, Loader=yaml.loader.SafeLoader)
    return metrics_data


def best_epoch_metrics(
    metrics: dict
) -> dict:
    """
    From the metrics dictionnary, returns only the best epoch metrics
    """
    val_metrics = {key: values[VALMETRIC] for key, values in metrics.items()}
    best_epoch = min(val_metrics, key=val_metrics.get)
    best_metrics = metrics[best_epoch]
    best_metrics["best_epoch"] = best_epoch
    return best_metrics


def get_config_params(
    path_to_expdir: Path
) -> dict:
    """
    Get config parameters from the yaml file
    """
    with open(path_to_expdir / "config.yaml", "r") as file:
        config = yaml.load(file, Loader=yaml.loader.SafeLoader)
    return config


def gridsearch_params(
    config: dict
) -> dict:
    """
    From config file, only keep the parameters used for gridsearch
    """
    gs_params = {
        'duration': config['task']['duration'],
        'batch_size': config['task']['batch_size'],
        'hidden_size': config['architecture']['lstm']['hidden_size'],
        'num_layers': config['architecture']['lstm']['num_layers'],
        'dropout': config['architecture']['lstm']['dropout'],
        'stride': config['architecture']['sincnet']['stride']
    }
    return gs_params


def append_data_dict(
    to_add: dict,
    original: dict = None
) -> dict:
    """
    Append the values of original dict with the values of to_add dict
    If original dict == None, return to_add, with values as list instead of scalars
    """
    if not original:
        dict_list = dict()
        for key, values in to_add.items():
            dict_list[key] = [values]
        return dict_list
    else:
        assert to_add.keys() == original.keys()
        for key, values in to_add.items():
            original[key].append(values)
        return original


def create_dataframe(
    data: dict
) -> pd.DataFrame:
    """
    Create a dataframe from data and config data
    """
    data_df = pd.DataFrame.from_dict(data)
    data_df.rename(
        columns=MAPPER,
        inplace=True
    )
    return data_df


def parse_args(argv):
    """Parser"""
    parser = argparse.ArgumentParser(description='Gets data from tensorboard events file and'
                                                 'creates a dataframe and csv with all params and data')
    
    subparsers = parser.add_subparsers()

    prepare = subparsers.add_parser("prepare")
    prepare.set_defaults(func=prepare_func)
    prepare.add_argument('gridsearch_path', type=str,
                        help="Path to all models")
    prepare.add_argument('heldout_gs_path', type=str,
                        help="Path to heldout experiments directory")
    prepare.add_argument('csv_file', type=str,
                        help="csv file with info about each experiment(best epoch..)")
    

    get_data = subparsers.add_parser("get_data")
    get_data.set_defaults(func=retreive_data)
    get_data.add_argument('gridsearch_path', type=str,
                        help="Path to all models")
    get_data.add_argument('--output', type=str, default="gridsearch_models_data.csv",
                        help="Path to csv file where all data are written. default: 'gridsearch_models_data.csv'")
  
    return parser.parse_args(argv)


def prepare_func(args):
    print("Preparing data")

    data_file = pd.read_csv(args.csv_file)

    for row in data_file.iterrows():
        experiment = row[1]['name']
        best = row[1]['best_epoch']
        prepare_data_dir(
            args.heldout_gs_path,
            experiment,
            best,
            args.gridsearch_path
        )


def retreive_data(args):
    print("Retrieving the data")

    all_models_data = None
    no_checkpoints = []

    experimental_dirs = Path(args.gridsearch_path).glob('*')
    for exp_dir in experimental_dirs:
        try:
            # get metrics
            if (exp_dir / "metrics.yaml").is_file():
                metrics = get_metrics_data(exp_dir / "metrics.yaml")
            else:
                metrics = dict()
                metrics = get_metrics(exp_dir)
                save_data(metrics, exp_dir)
            
            # get config params
            config = get_config_params(exp_dir)
            config = gridsearch_params(config)

            data = {**config, **metrics}
            data['name'] = exp_dir.name
            data['only_vad'] = 'only_vad' in data['name']
            all_models_data = append_data_dict(data, all_models_data)
        except FileNotFoundError:
            no_checkpoints.append(exp_dir)
    
    models_data_df = create_dataframe(all_models_data)
    models_data_df.to_csv(args.output)

    print("A problem occured for the following experiments : ")
    print(no_checkpoints)



if __name__ == "__main__":
    argv = sys.argv[1:]
    args = parse_args(argv)

    if hasattr(args, "func"):
        # print(args.func)
        args.func(args)
    else:
        print("Something went wrong with the parser")