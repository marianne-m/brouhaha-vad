#!/bin/bash
#SBATCH --account=xdz@v100
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1                # nombre de GPU a reserver
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread


python main.py runs/brouhaha/ train \
-p Brouhaha.SpeakerDiarization.NoisySpeakerDiarization \
--classes brouhaha \
--model_type pyannet \
--epoch 100 \
--data_dir "/gpfsscratch/rech/xdz/uzm31mf/final_dataset/NoisyDataset"
