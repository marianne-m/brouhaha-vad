import os
from tqdm import tqdm
import shutil

def split_data(audio_path,rttm_path,detailed_snr_labels_path,reverb_labels_path,save_path,percent_split):
  
  audios_names = os.listdir(audio_path)

  n_audios = len(audios_names)

  print(n_audios, ' audios')

  n_audios_test = int((n_audios * percent_split)/100)

  print(n_audios_test, ' para teste')

  #audios_test_names = [audios_names[i].replace('.wav','') for i in range(n_audios_test)] # trocar para flac
  audios_test_names = [audios_names[i].replace('.flac','') for i in range(n_audios_test)]

  for name in tqdm(audios_test_names):
    #move audios to new folder
    shutil.move(os.path.join(audio_path,name + '.flac'), os.path.join(save_path,'audio_16k' + '/' + name + '.flac'))

    #move rttm files to new folder
    shutil.move(os.path.join(rttm_path,name + '.rttm'), os.path.join(save_path,'rttm_files' + '/' + name + '.rttm'))

    #move snr_labels to new folder
    shutil.move(os.path.join(detailed_snr_labels_path,name + '.npy'), os.path.join(save_path,'detailed_snr_labels' + '/' + name + '.npy'))

    #write new file(test) for reverb_labels and erase in old file(train)
  
  print('\n write files...')

  with open(reverb_labels_path,"r") as oldfile:
    content = oldfile.readlines()
    oldfile.close()

  with open(os.path.join(save_path,'reverb_labels.txt'), "w") as newfile:
    for i in range(len(content)):
      if content[i].split()[0] in audios_test_names:
        newfile.write(content[i])

    newfile.close()

  with open(reverb_labels_path, "w") as oldfile:
    for i in range(len(content)):
      if content[i].split()[0] not in audios_test_names:
        oldfile.write(content[i])
    oldfile.close()

  print('---Processo concluido---')

def main():
  audio_path = input('Digite o caminho para os audios: ')
  rttm_path = input('Digite o caminho para os rttm: ')
  detailed_snr_labels_path = input('Digite o caminho para os detailed_snr_labels: ')
  reverb_labels_path = input('Digite o caminho para o reverb_labels: ')
  save_path = input('Digite o caminho para a pasta que deseja salvar o teste: ')
  percent_split = int(input('Digite a porcentagem para o teste: '))

  split_data(audio_path,rttm_path,detailed_snr_labels_path,reverb_labels_path,save_path,percent_split)

main()