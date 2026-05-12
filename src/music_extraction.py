import pandas as pd
import yt_dlp
import os
import re

CSV_FILE = 'data/music_views.csv' 
SAVE_PATH = 'data/music_data'

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def clean_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", str(filename)).strip()

def main():
    try:
        df = pd.read_csv(CSV_FILE)
        
        print(f"인식된 열 이름: {list(df.columns)}")
        
        total_count = len(df)
        print(f"총 {total_count}개의 곡을 처리합니다.")

        for i in range(total_count):
            row = df.iloc[i]
            title = row['title']
            url = row['titleUrl']

            video_id = url.split('v=')[-1].split('&')[0]
            
            safe_title = clean_filename(title)

            final_filename = f"{safe_title} [{video_id}].%(ext)s"
            
            current_ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(SAVE_PATH, final_filename), 
                'ignoreerrors': True,
                'quiet': True,
                'no_warnings': True,
            }
            
            print(f"[{i+1}/{total_count}] 다운로드 중: {safe_title} (ID: {video_id})")

            try:
                with yt_dlp.YoutubeDL(current_ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e:
                print(f"{safe_title} 다운로드 실패: {e}")

        print("\n모든 작업이 완료되었습니다!")

    except Exception as e:
        print(f"전체 프로세스 오류 발생: {e}")

if __name__ == "__main__":
    main()