import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks, butter, filtfilt
import mediapipe as mp
from utils import face_mesh_to_array, get_bbox, get_square_bbox  # utils.py에서 필요한 함수들을 가져옵니다.
from scipy import signal

# 미분 계산을 위한 함수
def derivative(signal):
    return np.gradient(signal)

# 밴드패스 필터 설정
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs = 30, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 그린 채널 신호 추출 함수
def extract_green_channel_signal(video_data):
    green_channel = video_data[:, :, :, 1]  # 그린 채널 추출
    # 각 프레임에 대한 그린 채널의 평균값 계산
    green_channel_mean = green_channel.mean(axis=(1, 2))
    detrended_signal = signal.detrend(green_channel_mean)
    smoothed_wave = gaussian_filter(detrended_signal, sigma=4)
    # 신호에 필터 적용
    filtered_signal = butter_bandpass_filter(smoothed_wave, 0.67, 3.33, fs=30, order=2)

    diff_smoothed_wave = derivative(filtered_signal)

    return diff_smoothed_wave


# 심박수 계산 함수
def calculate_hr(video_data, fps, window_size=150, step_size=30):
    rPPG_Signal = extract_green_channel_signal(video_data)
    bpm_per_frame = []
    times = []

    for start in range(0, len(rPPG_Signal) - window_size, step_size):
        end = start + window_size
        segment = rPPG_Signal[start:end]
        peaks, _ = find_peaks(segment, distance=fps//2, height=None)
        # print("Peaks found:", peaks)

        if len(peaks) > 1:
            ibi = np.diff(peaks) / fps
            bpm = 60 / np.mean(ibi) if len(ibi) > 0 else np.nan
        else:
            bpm = np.nan
        # print("BPM calculation:", bpm)

        bpm_per_frame.append(bpm)
        times.append((start + end) / 2 / fps)

    return times, bpm_per_frame

# rPPG 신호 추출 및 시각화
def plot_rPPG_signal(video_data):
    rPPG_Signal = extract_green_channel_signal(video_data)
    plt.figure(figsize=(10, 4))
    plt.plot(rPPG_Signal, label='rPPG Signal')
    plt.title('Extracted rPPG Signal Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()

# rPPG 신호 추출 및 시각화
def full_plot_rPPG_signal(video_data):
    rPPG_Signal = extract_green_channel_signal(video_data)
    plt.figure(figsize=(10, 4))
    plt.plot(rPPG_Signal, label='Full rPPG Signal')
    plt.title('Full Wave Signal')
    plt.xlabel('Frame Number')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()

def calculate_SI(signal, fps, window_size=150, step_size=30):
    # 1. rPPG 신호 추출
    rPPG_Signal = extract_green_channel_signal(signal)
    rppg = rPPG_Signal - np.mean(rPPG_Signal)  # DC 컴포넌트 제거

    # 2. PPI (Peak-to-Peak Interval) 추출
    peaks, _ = find_peaks(rppg, distance=fps//2)  # 예: 20은 최소 간격 (ms)입니다.
    nn_intervals = np.diff(peaks) / fps  # PPI를 초 단위로 변환

    # 3. Mode (Mo) 및 Amplitude of Mode (AMo) 계산
    bin_width = 0.05  # 50ms를 초 단위로 변환
    hist, bin_edges = np.histogram(nn_intervals, bins=np.arange(0, np.max(nn_intervals), bin_width))
    print(f"hist : {hist}")
    print(f"bin: {bin_edges}")
    Mo = bin_edges[np.argmax(hist)]
    AMo = (np.sum((nn_intervals >= Mo - bin_width/2) & (nn_intervals < Mo + bin_width/2)) / len(nn_intervals))

    # 4. Variation range (MxDMn) 계산
    Mx = np.max(nn_intervals)
    Mn = np.min(nn_intervals)   
    MxDMn = Mx - Mn

    # 5. Baevsky stress index (SI) 계산 0~40 사이 값으로 나옴
    SI = (AMo * 100) / (2 * Mo * MxDMn)
    return SI

# 메인 함수
def main():
    cap = cv2.VideoCapture(0)  # 웹캠 인덱스 확인 필요
    fps = cap.get(cv2.CAP_PROP_FPS)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    video_data = []
    full_video_data = []
    target_size = (64, 64)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            lmrks = face_mesh_to_array(results, frame.shape[1], frame.shape[0])

            if lmrks is not None:
                bbox = get_square_bbox(get_bbox(lmrks, frame.shape[1], frame.shape[0]), frame.shape[1], frame.shape[0])
                x1, y1, x2, y2 = bbox
                cropped = frame_rgb[y1:y2, x1:x2]
                resized = cv2.resize(cropped, target_size)
                video_data.append(resized)
                full_video_data.append(resized)

            if len(video_data) >= 151:
                video_array = np.array(video_data)
                plot_rPPG_signal(video_array)
                times, bpm_per_frame = calculate_hr(video_array, fps)
                print("Current heart rate measurements:", bpm_per_frame)
                video_data.pop(0)  # Maintain sliding window

            cv2.imshow('Webcam Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        full_video_data_array = np.array(full_video_data)
        cap.release()
        cv2.destroyAllWindows()
        # print(full_video_data_array)
        # full_plot_rPPG_signal(full_video_data_array)
        stress_index = calculate_SI(full_video_data_array, fps)
        print(f"stress index : {stress_index}")

if __name__ == "__main__":
    main()
    