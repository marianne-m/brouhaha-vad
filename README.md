# brouhaha-vtc

A modified pyannote VTC task and model for predicting SNR and reverberation (C50)


## Installation

```
git clone ssh://git@gitlab.cognitive-ml.fr:1022/htiteux/brouhaha-vtc.git
conda env create -f environment.yml
conda activate brouhaha-vad
pip install git+ssh://git@gitlab.cognitive-ml.fr:1022/htiteux/pyannote-brouhaha-db.git
```

## Specifying your database

Your database must have a train/dev/test split. Each set has the following structure :

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
│   └── ...
```

Define your database in the `~/.pyannote/database.yml` file with this line :

```
Databases:
  Bouhaha: Path/to/your/database
```


## Training

To train the model, use the following command :

```
python main.py runs/brouhaha/ train \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --classes brouhaha \
    --model_type pyannet \
    --epoch NB_OF_EPOCH_MAX \
    --data_dir "path/to/your/database"
```