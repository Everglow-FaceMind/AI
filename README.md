# To Do
**Purpose: 비디오를 통해 심박수 측정**
- 신호처리 기반 심박수 측정(Task1)
  - real time으로 사용자 얼굴 옆에 심박수가 위치 &rarr; 신호처리
- 딥러닝 기반 심박수 측정(Task2)
  - 3D CNN사용해서 실험
  - 2D & RNN(LSTM)사용해서 실험 

- 성능 개선을 위한 공통작업
  - ROI 수정
  - signal에서 peak 탐지시 적절한 dsitance 값 설정

- 작업
  - 실시간을 목표로 신호처리 기반 방법을 사용해 성능 개선
  - 딥러닝 기반 심박수 측정도 계속 시도(성능이 많이 좋다면 녹화 후 딥러닝 모델을 통해 심박수를 추론하도록 기능 수정)
# Task1
신호처리 기반 심박수 측정

# Task2
딥러닝 기반 심박수 측정

## 성능 개선
`/load_dataset/load_dataset_UBFC.ipynb` &rarr; UBFC 데이터셋에서 비디오를 얼굴 랜드마크 기준으로 crop 하고 64 x 64 resize 후 npz format(비디오, 랜드마크, 맥박신호, 심박수)으로 저장

모델 input 사이즈, ROI 등을 수정하려면 utils.py에서 수정
