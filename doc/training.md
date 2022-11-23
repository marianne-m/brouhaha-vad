## Specifying your database

First, you must install the pyannote-brouhaha-db package which contains all the necessary instructions to loop through Brouhaha data:

```bash
conda activate brouhaha
pip install https://github.com/marianne-m/pyannote-brouhaha-db.git
```

Your database must have a train/dev/test split. Each set must have the following tree structure :

```
train/
├── audio_16k
│   └── file_1.flac
│   └── ...
├── detailed_snr_labels
│   └── file_1_snr.npy
│   └── ...
├── reverb_labels.txt
└── rttm_files
    └── file_1.rttm
    └── ...
```

Define your database in the `~/.pyannote/database.yml` file with this line :

```
Databases:
  Brouhaha: Path/to/your/database/*/audio_16k/{uri}.flac
```


## Training

To train the model, use the following command :

```
python main.py train /path/to/experimental/directory \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --model_type pyannet \
    --epoch 35 \
    --data_dir path/to/your/database
```

#### Use a config.yaml

You can train your model with specific model hyper-parameters and specific task parameters. 
To do so, put a `config.yaml` in your experimental directory, as the following one :

```
task:
  duration: 2.0
  batch_size: 64
  lambda_vad: 1
architecture:
  sincnet:
    stride: 10
    sample_rate: 16000
  lstm:
    hidden_size: 128
    num_layers: 2
    bidirectional: true
    monolithic: true
    dropout: 0.0
    batch_first: true
  linear:
    hidden_size: 128
    num_layers: 2
```

And use the `--config` command when launching the training :

```
python main.py train /path/to/experimental/directory \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --model_type pyannet \
    --epoch 35 \
    --data_dir path/to/your/database \
    --config
```

## Tuning

Once the model is trained, you can tune the threshold for the voice activity detection with the following command :

```
python main.py tune /path/to/experimental/directory \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --model_path path/to/the/model/checkpoint \
    --data_dir path/to/your/database \
    --params path/to/best/params/yaml/file
```

If the `--params` flag is not specified, the best parameters will be saved in the file `best_params.yml` in the experimental
directory.


## Apply the model

You can apply the model on the test part of your pyannote database like this : 
```
python main.py apply \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --model_path path/to/the/model/checkpoint \
    --data_dir path/to/your/database \
    --out_dir path/to/the/inference/output/folder \
    --ext wav \
    --params path/to/best/params/yaml/file
```

If the `--params` is not specified, the parameters tuned on the Brouhaha database will be used.


## Score the model

Finally, you can score your model and compute the F-Score for the Voice Activity Detection, the Mean Squared Error for the
SNR and the Mean Squared Error for the C50.

```
python main.py score \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --model_path path/to/the/model/checkpoint \
    --data_dir path/to/your/database \
    --out_dir path/to/the/inference/output/folder \
    --report_path path/to/score/files
```
