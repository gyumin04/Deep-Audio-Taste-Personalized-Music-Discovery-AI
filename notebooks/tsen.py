import torch
import numpy as np
import os
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.model import AudioEmbedder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioEmbedder().to(device)
model.load_state_dict(torch.load('models/my_music_ai_model.pth'))
model.eval()

data_path = 'data/processed_data'
file_list = sorted(os.listdir(data_path))

all_embeddings = []

with torch.no_grad():
    for file_name in file_list:
        spec = np.load(os.path.join(data_path, file_name))
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0).unsqueeze(0).to(device)
        
        embedding = model(spec_tensor)
        all_embeddings.append(embedding.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)

NPY_DIR = 'data/processed_data' 

df = pd.read_csv('data/mapping.csv') 

df['full_path'] = df['id'].apply(lambda x: os.path.join(NPY_DIR, str(x) + '.npy'))

df = df[df['full_path'].apply(os.path.exists)]

pos_files = df[df['count'] >= 10]['id'].to_list()

my_fav_list = df[df['count'] >= 10]['id'].apply(lambda x: str(x) + '.npy').to_list()

pos_indices = []
neg_indices = []

for i, file_name in enumerate(file_list):
    if file_name in my_fav_list:
        pos_indices.append(i)
    else:
        neg_indices.append(i)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
vis_dims = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(10, 7))
plt.scatter(vis_dims[neg_indices, 0], vis_dims[neg_indices, 1], c='blue', alpha=0.2, label='Others')
plt.scatter(vis_dims[pos_indices, 0], vis_dims[pos_indices, 1], c='red', alpha=0.8, label='My Favorites')
plt.legend()
plt.show()