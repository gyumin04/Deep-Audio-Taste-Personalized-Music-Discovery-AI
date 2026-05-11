import librosa
import numpy as np
import os
import pandas as pd

def preprocess_audio(file_path, target_shape=(128, 128)):
    try:
        y, sr = librosa.load(file_path, sr=22050, offset=30, duration=20)
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_shape[0])
        
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        if S_dB.shape[1] > target_shape[1]:
            S_dB = S_dB[:, :target_shape[1]]
        else:
            S_dB = np.pad(S_dB, ((0, 0), (0, target_shape[1] - S_dB.shape[1])))
            
        return S_dB
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

save_dir = 'data/processed_data'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def process_and_save(file_path, song_id):
    spec = preprocess_audio(file_path)
    
    if spec is not None:
        save_path = os.path.join(save_dir, f"{song_id}.npy")
        np.save(save_path, spec)
        return save_path
    return None

mapping_data = pd.read_csv("data/mapping.csv")
file = mapping_data['file'].to_list()
count = mapping_data['count'].to_list()
id = mapping_data['id'].to_list()

for i in range(len(file)):
    process_and_save("data/music_data/" + file[i], id[i])