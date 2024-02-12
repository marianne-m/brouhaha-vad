from librosa import get_duration
import os

audios_path = '/home/deivison/workspace/VAD/DALI_database/dev/audio_16k'
annotated_path = '/home/deivison/workspace/VAD/DALI_database/dev/dev.uem'


list_dir = os.listdir(audios_path)

with open(annotated_path, "w") as annotated_file:
    for i in list_dir:

        name_file  = i.replace('.flac','')
        end = get_duration(path = audios_path + '/' + i)

        annotated_file.write("%s\n" % f'{name_file} NA 0.000 {end}')

    annotated_file.close()