import numpy as np
from scipy.signal import find_peaks

"""
Baevsky의 스트레스 지수(SI)

Amo (Amplitude Mode) : 심박간격(HR)의 변동성을 나타내는 지표
Modulus (Mo) : 심박간격의 평균값
M : Mo와 관련된 보정 계수
DMn (Day-Mean): 일일 평균 심박 주기 간격의 변동성
"""



def calculate_SI(npz_data, fps=fps, window_size=150, step_size=30):
  PPG_Signal = npz_data['wave']
  ppg_signal = PPG_Signal - np.mean(PPG_Signal)  # DC 컴포넌트 제거

  # 2. NN 간격 추출
  # PPG 신호에서 NN 간격 추출
  peaks, _ = find_peaks(ppg_signal, distance=10)  # 예: 20은 최소 간격 (ms)입니다.
  nn_intervals = np.diff(peaks)  # NN 간격 계산

  # 3. AMo, Mo, MxDMn 계산
  amplitude_mode = np.max(np.histogram(nn_intervals, bins=np.arange(0, 2000, 50))[0])  # AMo 계산
  mode_mo = np.median(nn_intervals)  # Mo 계산
  dmn = np.std(nn_intervals)  # DMn 계산

  # 4. SI 계산
  M = 1  # 보정 계수 (실제 값에 따라 조절 필요)
  stress_index = (amplitude_mode * 100) / (2 * mode_mo * M * dmn)

  print(f"AMo: {amplitude_mode}, Mo: {mode_mo}, MxDMn: {dmn}")
  print(f"Stress Index (SI): {stress_index}")
