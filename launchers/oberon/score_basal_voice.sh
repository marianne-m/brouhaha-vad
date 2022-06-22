#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodelist=puck5
#SBTACH --gres=gpu:rtx8000:1

# loading modules and activating the right conda env
source ~htiteux/.bashrc
module load anaconda espeak mbrola
conda activate pyannote-vtc-v2

python main.py runs/basal_voice/ score \
-p BasalVoice.SpeakerDiarization.InterviewDiarizationProtocol \
--model_path runs/basal_voice/checkpoints/last.ckpt \
--classes basal_voice \
--metric ier \
--apply_folder runs/basal_voice/apply_ier/ \
--report_path runs/baal_voice/results/ier_tuned_ier.csv
