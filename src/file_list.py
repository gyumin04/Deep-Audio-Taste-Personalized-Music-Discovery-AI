import os
import pandas as pd

path = "data/music_data"
file_list = os.listdir(path)

data = pd.read_csv('data/music_views.csv')

count = data['count'].to_list()
url_id = data['titleUrl'].to_list()

final_id = []

for id in url_id:
    final_id.append(id.replace("https://music.youtube.com/watch?v=", ""))

mapping = {
    "file" : [],
    "count" : [],
    'id' : []
}

for i in range(len(file_list)):
    for j in range(len(final_id)):
        if final_id[j] in file_list[i]:
            mapping['file'].append(file_list[i])
            mapping['count'].append(count[j])
            mapping['id'].append(final_id[j])

mapping = pd.DataFrame(mapping)

mapping.to_csv("data/mapping.csv")