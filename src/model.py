import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.nn as nn

NPY_DIR = 'data/processed_data' 

df = pd.read_csv('data/mapping.csv') 

df['full_path'] = df['id'].apply(lambda x: os.path.join(NPY_DIR, str(x) + '.npy'))

df = df[df['full_path'].apply(os.path.exists)]

pos_files = df[df['count'] >= 10]['full_path'].tolist()
neg_files = df[df['count'] < 10]['full_path'].tolist()

print(f"매칭 성공 - 선호 곡: {len(pos_files)}개 / 기타: {len(neg_files)}개")

class MusicTripletDataset(Dataset):
    def __init__(self, pos_files, neg_files, iterations=1000):
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.iterations = iterations

    def __len__(self):
        return self.iterations

    def __getitem__(self, idx):
        anc_path = np.random.choice(self.pos_files)
        pos_path = np.random.choice(self.pos_files)
        neg_path = np.random.choice(self.neg_files)

        anc = np.load(anc_path)[np.newaxis, ...]
        pos = np.load(pos_path)[np.newaxis, ...]
        neg = np.load(neg_path)[np.newaxis, ...]

        return torch.from_numpy(anc).float(), torch.from_numpy(pos).float(), torch.from_numpy(neg).float()

train_ds = MusicTripletDataset(pos_files, neg_files)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

class AudioEmbedder(nn.Module):
    def __init__(self):
        super(AudioEmbedder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioEmbedder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.TripletMarginLoss(margin=0.2)

for epoch in range(30):
    model.train()
    total_loss = 0
    for anc, pos, neg in train_loader:
        anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
        
        optimizer.zero_grad()
        a_emb, p_emb, n_emb = model(anc), model(pos), model(neg)
        loss = criterion(a_emb, p_emb, n_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} 완료. Loss: {total_loss/len(train_loader):.4f}")

model.eval()
all_embeddings = []
file_paths = df['full_path'].tolist()

with torch.no_grad():
    for path in file_paths:
        data = np.load(path)[np.newaxis, np.newaxis, ...]
        data_tensor = torch.from_numpy(data).float().to(device)
        embedding = model(data_tensor)
        all_embeddings.append(embedding.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)

pos_indices = df[df['count'] >= 10].index
target_vector = np.mean(all_embeddings[pos_indices], axis=0)

from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([target_vector], all_embeddings)[0]

df['similarity'] = similarities
recommendations = df[df['count'] < 5].sort_values(by='similarity', ascending=False)

print("--- 당신을 위한 AI 추천 곡 TOP 10 ---")
print(recommendations[['id', 'similarity']].head(10))

torch.save(model.state_dict(), 'models/my_music_ai_model.pth')
