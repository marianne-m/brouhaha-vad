#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=puck5
#SBTACH --gres=gpu:rtx8000:1

# loading modules and activating the right conda env
source ~htiteux/.bashrc
module load anaconda espeak mbrola
conda activate pyannote-vtc-v2

python main.py runs/basal_voice/ apply \
-p BasalVoice.SpeakerDiarization.InterviewDiarizationProtocol \
--model_path runs/basal_voice/checkpoints/last.ckpt \
--classes basal_voice \
--apply_folder runs/basal_voice/apply_ier/ \
--params runs/basal_voice/best_params_ier.yml
