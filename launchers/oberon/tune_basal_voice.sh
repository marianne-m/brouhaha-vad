#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=puck5
#SBTACH --gres=gpu:rtx8000:1

# loading modules and activating the right conda env
source ~htiteux/.bashrc
module load anaconda espeak mbrola
conda activate pyannote-vtc-v2

python main.py runs/basal_voice/ tune \
-p BasalVoice.SpeakerDiarization.InterviewDiarizationProtocol \
--model_path runs/basal_voice/checkpoints/last.ckpt \
-nit 50 \
--classes basal_voice \
--metric fscore
