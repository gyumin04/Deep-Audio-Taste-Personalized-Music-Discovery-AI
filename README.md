# Deep-Audio-Taste: Personalized Music Discovery AI
> YouTube 시청 기록을 기반으로 사용자의 음악 취향(Bass & Electric 사운드)을 학습하고 새로운 곡을 추천하는 딥러닝 프로젝트입니다.

---

## 프로젝트 개요
이 프로젝트는 단순한 장르 분류를 넘어, 사용자가 선호하는 음원의 **수학적 특징**(Embedding)을 추출합니다. **Siamese Network**와 **Triplet Loss**를 활용하여, 사용자가 자주 들은 곡들 사이의 거리는 좁히고, 그렇지 않은 곡과의 거리는 멀어지도록 학습되었습니다.

## 데이터셋 규모 (Dataset Scale)
본 프로젝트는 사용자의 실제 YouTube 음원 청취 데이터셋을 가공하여 활용하였습니다. 데이터 보안 및 저작권 준수를 위해 원본 오디오 파일은 저장소에서 제외되었습니다.

- **전체 오디오 데이터:** 총 1,382곡 (MP3 포맷)
- **사용자 선호 데이터 (Positive):** 청취 횟수 10회 이상의 핵심 선호 곡 72곡
- **비선호/일반 데이터 (Negative):** 선호 데이터를 제외한 무작위 1,310여 곡
- **데이터 전처리 규모:** 각 음원을 `Librosa`를 통해 멜-스펙트로그램(Mel-Spectrogram) 행렬 데이터(`.npy`)로 변환 및 임베딩 추출 (총 1,382개 파일 생성)

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
  
## 모델 아키텍처 및 학습 파라미터 (Model Architecture & Training)
프로젝트에 사용된 딥러닝 모델의 세부 구조와 하이퍼파라미터 설정은 다음과 같습니다.

### 1. Model Architecture
- **Base Network:** 멜-스펙트로그램(Mel-Spectrogram)의 오디오 특징 추출을 위해 1D/2D Convolutional Layer와 합성곱 블록(CNN)을 조합한 Custom 아키텍처 설계
- **Embedding Layer:** 고차원 오디오 특징을 최종적으로 **128차원의 밀집 벡터(Dense Embedding)**로 매핑
- **Sub-Network:** 두 개의 오디오 인풋이 동일한 가중치를 공유하는 **Siamese Network** 구조 적용

### 2. Hyperparameters & Training Details
- **Loss Function:** Triplet Margin Loss (`margin=1.0`) 적용
  - *Positive(선호 곡) 간의 거리는 최소화하고, Negative(일반 곡)와의 거리는 최소 1.0 이상 확보하도록 학습*
- **Optimizer:** Adam (Learning Rate: `1e-4`)
- **Batch Size:** 16
- **Input Shape:** `(Batch_Size, 1, 128, 128)`

## 시각화 결과 (Analysis)
<img width="1000" height="700" alt="Success of Taste Embedding" src="https://github.com/user-attachments/assets/441cc497-6493-4817-a0ae-dc096f06545e" />
학습된 모든 음악 데이터를 2차원 공간에 투영한 결과입니다. 빨간색 점(선호 곡)들이 특정 영역에 밀집되어 있는 것을 확인할 수 있으며, 이는 모델이 사용자의 일관된 취향을 성공적으로 포착했음을 보여줍니다.

## 추천 결과 및 정성적 평가
기존에 알지 못했던 아티스트의 일렉트로닉 장르 곡이 취향 중심점과 매우 가깝게 추천되었으며, 

실제 청취 결과 개인 선호 기준(베이스 리프 중심)에 높은 일치도를 확인했습니다. 

또한, 제목과 업로더가 달라 필터링하기 힘들었던 중복 음원을 98% 이상의 유사도로 효과적으로 식별해내는 부가적 성과를 거두었습니다.

## 폴더 구조
```text
├── data/               # 데이터셋 (저작권 문제로 미포함)
├── models/             # 학습된 모델 가중치 (.pth)
├── notebooks/          # 데이터 분석 및 시각화 과정 (Jupyter Notebook)
├── src/                # 전처리, 모델 정의, 학습 및 추천 스크립트
└── requirements.txt    # 의존성 라이브러리 목록
