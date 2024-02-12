import shutil
from pytube import YouTube
import os
from pydub import AudioSegment
from pytube import Playlist
import re
import DALI as dali_code
from tqdm import tqdm
import librosa
from scipy.io.wavfile import write

def sr_conversion(input_filepath,output_filepath,target_sr):

    waveform, sr = librosa.load(input_filepath, sr=None)
    print(sr)
    target_waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
    write(output_filepath, target_sr, target_waveform)
    print(f'SAMPLING RATE CHANGED TO {target_sr}')

def write_rttm(dali_data, output_path,music_id):

  music_time = dali_data[music_id].annotations['annot']['lines']

  with open(os.path.join(output_path,music_id + '.rttm'), 'wb') as f:
      for turn in music_time:
          fields = ['SPEAKER', str(music_id), '1', str(round(turn['time'][0],3)), str(round(turn['time'][1] - turn['time'][0],3)),
                '<NA>', '<NA>', 'A', '<NA>', '<NA>']
          line = ' '.join(fields)
          f.write(line.encode('utf-8'))
          f.write(b'\n')

def download_music_youtube(VIDEO_URL,path, name):

  yt = YouTube(VIDEO_URL)

  title = yt.title

  audio = yt.streams.filter(only_audio=True)[0]
  audio.download(path,filename= name + '.mp4')

  new_filename= name +'.m4a'
  default_filename = name +'.mp4'

  shutil.move(os.path.join(path,default_filename),os.path.join(path, new_filename))

  # m4a to flac
  m4a_file = os.path.join(path, new_filename)
  flac_file = os.path.join(path, name + '.flac')
  sound = AudioSegment.from_file(m4a_file)
  sound.export(os.path.join(path, name + '.flac'),format = "flac")

  #wav_file = os.path.join(path, name + '.wav')
  #sound = AudioSegment.from_file(os.path.join(path, new_filename))
  #sound.export(wav_file, format="wav")

  os.remove(m4a_file)

  #change sample rate to 16K
  target_sr = 16000
  sr_conversion(flac_file,flac_file,target_sr)

  print(name)
  print('== Download Completo ==')

  return name

def load_dali(dali_data_path):
  dali_data = dali_code.get_the_DALI_dataset(dali_data_path, skip=[], keep=[])

  dali_info = dali_code.get_info(dali_data_path + '/' +  'info/DALI_DATA_INFO.gz')

  DALI_SAMLPES = []

  for i in dali_info:
    if i[3] == 'True':
      #if len(DALI_SAMLPES) < 3: #tirar restrição
      DALI_SAMLPES.append(i)

  DALI_SAMPLES_link = [i[2] for i in  DALI_SAMLPES]

  return dali_data, DALI_SAMLPES, DALI_SAMPLES_link

def main(dali_data_path, path, path_save_rttm):
  print('data loading...')
  dali_data,DALI_SAMLPES,_ = load_dali(dali_data_path)

  for sample in tqdm(DALI_SAMLPES):
    try:
      download_music_youtube('https://www.youtube.com/watch?v=' + sample[2], path, sample[0])
      write_rttm(dali_data, path_save_rttm,sample[0])
    except:
      continue

dali_data_path = input('digite o caminho do dali_dataset: ')
path = input('caminho da pasta para musicas: ')
path_save_rttm = input('caminho da pasta para rttm: ')

main(dali_data_path,path,path_save_rttm)