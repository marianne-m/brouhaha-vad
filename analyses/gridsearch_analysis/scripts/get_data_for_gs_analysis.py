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
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import yaml
import sys
from collections import defaultdict
from pathlib import Path
from typing import Union



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



def get_data_from_eventfile(
    file: Union[Path, str]
) -> dict:
    """
    Get training losses, validation metrics from tensorboard events file

    Returns
    -------
    rearranged_data : dict
        Dictionnary with epochs as keys and the values are a dictionnary 
        with metric name as keys and their values as values
    """
    data = defaultdict()

    # create the dict with step keys
    summary_it = tf.compat.v1.train.summary_iterator(file)
    next(summary_it)
    for e in summary_it:
        if e.summary.value[0].tag in METRICS:
            data[e.step] = defaultdict()

    # populate the dict
    summary_it = tf.compat.v1.train.summary_iterator(file)
    next(summary_it)
    for e in summary_it:
        if e.summary.value[0].tag in METRICS:
            data[e.step][e.summary.value[0].tag] = e.summary.value[0].simple_value

    rearranged_data = dict()
    for values in data.values():
        epoch = values.pop('epoch')
        rearranged_data[epoch] = values
    
    return rearranged_data


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
    path_to_expdir: Path
) -> dict:
    """
    Save data to a yaml file
    """
    with open(path_to_expdir / "metrics.yaml", "r") as file:
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

    parser.add_argument('gridsearch_path', type=str,
                        help="Path to all models")
    parser.add_argument('--output', type=str, default="gridsearch_models_data.csv",
                        help="Path to csv file where all data are written. default: 'gridsearch_models_data.csv'")       

    return parser.parse_args(argv)


def main(argv):
    """
    Look into args.gridsearch_path and for each model, gets the metrics either
    from metrics.yaml if it exists, or from the tensorboard events file.
    Then keep the best epoch metrics and the relevent parameters for the gridsearch
    analysis, and put everything into a csv file
    """
    args = parse_args(argv)

    all_models_data = None
    no_checkpoints = []

    experimental_dirs = Path(args.gridsearch_path).glob('*')
    for exp_dir in experimental_dirs:
        # get metrics
        try:
            if (exp_dir / "metrics.yaml").is_file():
                metrics = get_metrics_data(exp_dir / "metrics.yaml")
            else:
                metrics = dict()
                event_files = [file for file in (exp_dir / "VADTest").glob("events*")]
                for file in event_files:
                    metrics = {**metrics, **get_data_from_eventfile(str(file))}
                save_data(metrics, exp_dir)

            best_metrics = best_epoch_metrics(metrics)
            
            # get config params
            config = get_config_params(exp_dir)
            config = gridsearch_params(config)

            data = {**config, **best_metrics}
            data['name'] = exp_dir.name
            data['only_vad'] = 'only_vad' in data['name']
            all_models_data = append_data_dict(data, all_models_data)
        except ValueError:
            print(f'No checkpoint nor logs for {exp_dir.name}')
            no_checkpoints.append(exp_dir.name)
    
    models_data_df = create_dataframe(all_models_data)
    models_data_df.to_csv(args.output)

    print("A problem occured for the following experiments : ")
    print(no_checkpoints)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)