import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import mediapipe as mp
from utils import face_mesh_to_array, get_bbox, get_square_bbox  # utils.py에서 필요한 함수들을 가져옵니다.

# 미분 계산을 위한 함수
def derivative(signal):
    return np.gradient(signal)

# 그린 채널 신호 추출 함수
def extract_green_channel_signal(video_data):
    green_channel = video_data[:, :, :, 1]  # 그린 채널 추출
    green_channel_mean = green_channel.mean(axis=(1, 2))
    # print("Green channel mean:", green_channel_mean)
    smoothed_wave = gaussian_filter(green_channel_mean, sigma=2)
    diff_smoothed_wave = derivative(smoothed_wave)
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
    rPPG_Signal = extract_green_channel_signal(signal)
    rppg = rPPG_Signal - np.mean(rPPG_Signal)  # DC 컴포넌트 제거
    print(rppg.shape)
    # 2. NN 간격 추출
    # PPG 신호에서 NN 간격 추출
    peaks, _ = find_peaks(rppg, distance=fps//2)  # 예: 20은 최소 간격 (ms)입니다.
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

# 메인 함수
def main():
    cap = cv2.VideoCapture(1)  # 웹캠 인덱스 확인 필요
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
        full_plot_rPPG_signal(full_video_data_array)
        stress_index = calculate_SI(full_video_data_array, fps)
        print(stress_index)
if __name__ == "__main__":
    main()