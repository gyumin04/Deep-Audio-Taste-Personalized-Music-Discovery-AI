import pandas as pd

data = pd.read_json("data/시청 기록.json")
data = data[data["header"] == "YouTube Music"]

view = []
title = []
channel = []

music_list = data.drop_duplicates(subset=['titleUrl'])
music_list = music_list.reset_index(drop=True)
link = music_list['titleUrl']

for i in range(len(music_list["subtitles"])):
    channel.append(music_list['subtitles'][i][0]['name'])
music_list = music_list['title']


for i in range(len(music_list)):
    title.append(music_list[i].replace(" 을(를) 시청했습니다.", ""))
    view.append(len(data[data['titleUrl'] == link[i]]))

music_views = pd.DataFrame(columns=["title", 'channel', "count", 'titleUrl'])
music_views['title'] = title
music_views['count'] = view
music_views['channel'] = channel
music_views['titleUrl'] = link
music_views = music_views.sort_values(by='count', ascending=False)
music_views = music_views.reset_index(drop=True)

music_views.to_csv("data/music_views.csv")