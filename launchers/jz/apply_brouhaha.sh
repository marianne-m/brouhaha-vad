#!/bin/bash
#SBATCH --account=xdz@v100
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

BEST='best.ckpt'
MODEL="/gpfsscratch/rech/xdz/uzm31mf/Gridsearch_dev/dur_6_bs_64_lstm_hs_256_lstm_nl_3_dropout_0.5/checkpoints"

python main.py runs/brouhaha/ apply \
-p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
--model_path $MODEL/$BEST \
--classes brouhaha \
--data_dir "/gpfsscratch/rech/xdz/uzm31mf/final_dataset/NoisyDataset/" \
--set "test"
