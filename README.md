# To Do
**Purpose: 비디오를 통해 심박수 측정**
- 신호처리 기반 심박수 측정(Task1)
  - real time으로 사용자 얼굴 옆에 심박수가 위치 &rarr; 신호처리
- ~~딥러닝 기반 심박수 측정(Task2)~~
  - ~~3D CNN사용해서 실험~~
  - ~~2D & RNN(LSTM)사용해서 실험~~

- 성능 개선을 위한 공통작업
  - ROI 수정
  - 데이터 전처리(조명 정규화 등)
  - signal에서 peak 탐지시 적절한 dsitance 값 설정

- 작업
  - ~~실시간을 목표로 신호처리 기반 방법을 사용해 성능 개선~~
  - ~~딥러닝 기반 심박수 측정도 계속 시도(성능이 많이 좋다면 녹화 후 딥러닝 모델을 통해 심박수를 추론하도록 기능 수정)~~
  - 녹화방식이 아닌 실시간으로 프로젝트를 진행(딥러닝 사용불가)
  - 서버에서 영상을 프레임 단위로 들어오면 mediapipe를 통해 얼굴인식 후 crop, 프레임이 어느정도 쌓이면(윈도우 슬라이딩 값) 심박수를 계산해서 실시간으로 전송
    
# Task1
**신호처리 기반 심박수 측정**
1. ROI 설정(얼굴, 뺨, 이마 등)
2. RGB값 중 Green 채널의 평균값 변화를 신호로
3. Detrending(신호에서 저주파수 변동(예: 조명 변화, 느린 움직임 등)을 제거)
4. 가우시안 필터링(노이즈를 줄이고 신호의 품질 개선)
5. 밴드패스 필터링(주파수 범위: 0.67 Hz ~ 3.33 Hz)
6. Peak 탐지 후 심박수 측정

# ~~Task2~~
**~~딥러닝 기반 심박수 측정~~**

~~딥러닝 모델 성능 개선 방법~~
- ~~데이터 증강~~
- ~~3D CNN 모델 서치(현재 PhysNet사용, RPNet 사용예정)~~


## 성능 개선
`/load_dataset/load_dataset_UBFC.ipynb` &rarr; UBFC 데이터셋에서 비디오를 얼굴 랜드마크 기준으로 crop 하고 64 x 64 resize 후 npz format(비디오, 랜드마크, 맥박신호, 심박수)으로 저장

모델 input 사이즈, ROI 등을 수정하려면 utils.py에서 수정
