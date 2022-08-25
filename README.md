# brouhaha-vtc

A modified pyannote VTC task and model for predicting SNR and intelligibility (c50)


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
```

Create of modify your `~/.pyannote/database.yml` file with this line :

```
Databases:
  Bouhaha: Path/to/your/database
```


## Training

Launchers for Jean Zay are available in `launchers/jz`.

To train the model, use the following command :

```
python main.py runs/brouhaha/ train \
    -p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
    --classes brouhaha \
    --model_type pyannet \
    --epoch 10 \
    --data_dir "path/to/your/database"
```