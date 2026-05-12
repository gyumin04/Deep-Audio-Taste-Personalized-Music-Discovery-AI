# Deep-Audio-Taste: Personalized Music Discovery AI
> YouTube 시청 기록을 기반으로 사용자의 음악 취향(Bass & Electric 사운드)을 학습하고 새로운 곡을 추천하는 딥러닝 프로젝트입니다.

---

## 프로젝트 개요
이 프로젝트는 단순한 장르 분류를 넘어, 사용자가 선호하는 음원의 **수학적 특징**(Embedding)**을 추출합니다. **Siamese Network**와 **Triplet Loss**를 활용하여, 사용자가 자주 들은 곡들 사이의 거리는 좁히고, 그렇지 않은 곡과의 거리는 멀어지도록 학습되었습니다.

## 주요 기능
- **음원 전처리:** `Librosa`를 사용하여 MP3 파일을 멜-스펙트로그램(Mel-Spectrogram)으로 변환
- **취향 임베딩 학습:** Siamese Network 아키텍처를 통한 고차원 특징 추출
- **유사도 기반 추천:** 학습된 모델을 통해 1,400여 곡 중 사용자의 '취향 중심점'과 가장 가까운 곡 선정
- **중복 곡 식별:** 오디오 지문을 분석하여 제목이 달라도 음원이 같은 곡을 98% 이상의 유사도로 식별

## 기술 스택
- **Language:** Python 3.x
- **Deep Learning:** PyTorch
- **Audio Processing:** Librosa, Torchaudio
- **Data Analysis:** Pandas, Numpy, Scikit-learn
- **Visualization:** Matplotlib, T-SNE

## 시각화 결과 (Analysis)
<img width="1000" height="700" alt="Success of Taste Embedding" src="https://github.com/user-attachments/assets/441cc497-6493-4817-a0ae-dc096f06545e" />
학습된 모든 음악 데이터를 2차원 공간에 투영한 결과입니다. 빨간색 점(선호 곡)들이 특정 영역에 밀집되어 있는 것을 확인할 수 있으며, 이는 모델이 사용자의 일관된 취향을 성공적으로 포착했음을 보여줍니다.

## 폴더 구조
```text
├── data/               # 데이터셋 (저작권 문제로 미포함)
├── models/             # 학습된 모델 가중치 (.pth)
├── notebooks/          # 데이터 분석 및 시각화 과정 (Jupyter Notebook)
├── src/                # 전처리, 모델 정의, 학습 및 추천 스크립트
└── requirements.txt    # 의존성 라이브러리 목록
