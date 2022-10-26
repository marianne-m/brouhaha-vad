# Brouhaha: multi-task training for voice activity detection, speech-to-noise ratio, and C50 room acoustics estimation (2023)

![](doc/brouhaha.png)

Here's the companion repository of *Brouhaha*. 
You'll find the instructions to install and run our pretrained model. Given an audio segment, Brouhaha extracts:
- speech/non-speech segments.
- Speech-to-Noise Ratio (SNR) , that measures the speech level compared to the noise level.. 
- C50, that measures to which extent the environment is reverberant

You can listen to some audio samples we generated to train the model [here](https://marvinlvn.github.io/brouhaha/).

If you want to dig further, you'll also find the instructions to run the audio contamination pipeline, and retrain a model from scratch.

### Installation

```
git clone https://github.com/marianne-m/brouhaha-vad.git
cd brouhaha-vad
conda env create -f environment.yml
conda activate brouhaha-vad
conda install -c conda-forge libsndfile
```

### Extracting predictions

```
python main.py path/to/predictions apply \
          --model_path models/best/checkpoints/best.ckpt \
          --classes brouhaha \
          --data_dir path/to/data \
          --ext "wav"
```

### Going further

1) [Run the audio contamination pipeline](https://github.com/marianne-m/brouhaha-maker)
2) [Train your own model](./docs/training.md)