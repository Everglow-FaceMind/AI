# AI
`/load_dataset/load_dataset_UBFC.ipynb` &rarr; UBFC 데이터셋에서 비디오를 얼굴 랜드마크 기준으로 crop 하고 64 x 64 resize 후 npz format(비디오, 랜드마크, 맥박신호, 심박수)으로 저장

모델 input 사이즈, ROI 등을 수정하려면 utils.py에서 수정
