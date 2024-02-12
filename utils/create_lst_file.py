import os

fold_path = '/home/deivison/workspace/VAD/DALI_database/test/audio_16k'
lst_path = '/home/deivison/workspace/VAD/DALI_database/test/test.lst'


list_dir = os.listdir(fold_path)

new_list_dir = [i.replace('.flac','') for i in list_dir]

with open(lst_path, "w") as lst_file:
    for i in new_list_dir:
        lst_file.write("%s\n" % i)
    lst_file.close()