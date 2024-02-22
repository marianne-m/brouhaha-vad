import os
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from librosa import get_duration
from statistics import mean

folder_path_inference = input('inference_folder_path(.rttm): ')
folder_path_groundtruth = input('groundtruth_folder_path(.rttm): ')
folder_path_audio_gt = input('Audio folder path: ')
output_folder = input('output folder: ')

def get_frames(folder_path):
  svd_brouhaha_frames = {}

  if os.path.isdir(folder_path):
    files = os.listdir(folder_path)
    for file in files:
      file_path = f'{folder_path}/{file}'
      file_name = os.path.splitext(os.path.basename(file_path))[0]

      with open(file_path, "r") as f:
        text = f.readlines()

        if len(text) != 0:
          svd_brouhaha_frames[f'{file_name}'] = []

          for i in text:
            start = float(i.split(' ')[3])
            duration = float(i.split(' ')[4])
            end = start + duration

            svd_brouhaha_frames[f'{file_name}'].append({ 'start': start, 'end': end })
  else:
    print(f"The folder path '{folder_path}' does not exist.")
  
  return svd_brouhaha_frames

def extract_fscore_param(frames, duration, batch_size):
  fscore_array = []

  for i in range(int(duration / batch_size)):
    base_start = i * batch_size
    base_end = base_start + batch_size - .001

    entry = 0

    for frame in frames:
      if frame['start'] <= base_start and base_end <= frame['end']:
        entry = 1

    fscore_array.append(entry)


  return fscore_array

def run_fscore(music_name):
  duration = music_durations[f'{music_name}']
  batch_size = .160 #160ms

  y_true = extract_fscore_param(svd_dali_frames[f'{music_name}'], duration, batch_size)
  y_pred = extract_fscore_param(svd_brouhaha_frames[f'{music_name}'], duration, batch_size)


  return f1_score(y_true, y_pred, average='macro')

# Run a extraction of metrics 

fscores = []
music_durations = {}

if os.path.isdir(folder_path_audio_gt):
  folder_path = folder_path_audio_gt
  files = os.listdir(folder_path)
  for file in files:
    file_path = f'{folder_path}/{file}'
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    music_durations[f'{file_name}'] = get_duration(path=file_path)
    
svd_brouhaha_frames = get_frames(folder_path_inference)
svd_dali_frames = get_frames(folder_path_groundtruth)

for music_name in list(svd_brouhaha_frames.keys()):
  try:
    score = run_fscore(music_name)
    fscores.append(score)
  except:
    continue

# create and save plots

output_filename = output_folder + '/' + 'f1_score_model.png'

fig = plt.figure()
ax = fig.add_subplot(1,1,1,)
ax.hist(fscores, 20, rwidth=0.9, alpha=0.7, edgecolor='black')

plt.title('F1-score Distribution')
plt.xlabel('F1-socre')
plt.ylabel('Number of occurrences')
plt.savefig(output_filename)

# save mean f1-score
output_f1_mean = output_folder + '/' + 'f1_score_model.txt'

with open(output_f1_mean, "w") as f1_txt:
  f1_txt.write(f'Mean: {mean(fscores)}')

