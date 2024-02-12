import os

rttm_path = '/home/deivison/workspace/VAD/DALI_database/train/rttm_files'
rttm_reference_path = '/home/deivison/workspace/VAD/DALI_database/train/train.rttm'


list_dir = os.listdir(rttm_path)

with open(rttm_reference_path, "w") as rttm_reference_file:
    for i in list_dir:
        with open(rttm_path + '/' + i, "r") as rttm_file:
            lines = rttm_file.readlines()
            for j in lines:
                rttm_reference_file.write(j)
        rttm_file.close()
    rttm_reference_file.close()