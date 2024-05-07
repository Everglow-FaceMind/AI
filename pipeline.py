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
        peaks, _ = find_peaks(segment, distance=10, height=None)
        print("Peaks found:", peaks)

        if len(peaks) > 1:
            ibi = np.diff(peaks) / fps
            bpm = 60 / np.mean(ibi) if len(ibi) > 0 else np.nan
        else:
            bpm = np.nan
        # print("BPM calculation:", bpm)

        bpm_per_frame.append(bpm)
        times.append((start + end) / 2 / fps)

    return times, bpm_per_frame

# 메인 함수
def main():
    cap = cv2.VideoCapture(1)  # 기본 웹캠 사용
    fps = cap.get(cv2.CAP_PROP_FPS)  # FPS 추출
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    video_data = []
    target_size = (64, 64)  # 모든 이미지를 이 크기로 조정


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
                resized = cv2.resize(cropped, target_size)  # 이미지를 동일한 크기로 리사이징
                cv2.imwrite('imimg.jpg', resized)
                video_data.append(resized)


            if len(video_data) >= 150:  # 150 프레임 축적 후 심박수 계산
                if len(video_data) > 150:
                    video_data.pop(0)  # 가장 오래된 프레임 제거
                video_array = np.array(video_data)
                plot_rPPG_signal(video_array) 
                # print(video_array.shape)
                times, bpm_per_frame = calculate_hr(video_array, fps)
                print(bpm_per_frame)
                # plt.plot(times, bpm_per_frame)
                # plt.xlabel('Time (s)')
                # plt.ylabel('Estimated Heart Rate (BPM)')
                # plt.title('Heart Rate Over Time')
                # plt.show(block=False)
                # plt.pause(0.01)
                # plt.clf()

            cv2.imshow('Webcam Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
